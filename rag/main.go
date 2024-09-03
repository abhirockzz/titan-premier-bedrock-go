package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/jackc/pgx/v5"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/embeddings/bedrock"
	bedrock_llm "github.com/tmc/langchaingo/llms/bedrock"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/pgvector"
)

var store pgvector.Store
var llm *bedrock_llm.LLM

const modelID = "amazon.titan-text-premier-v1:0"
const maxTokenCountLimitForTitanTextPremier = 3072

func init() {

	host := "localhost"
	user := "postgres"
	password := "postgres"
	dbName := "postgres"

	connURLFormat := "postgres://%s:%s@%s:5432/%s?sslmode=disable"

	pgConnURL := fmt.Sprintf(connURLFormat, user, url.QueryEscape(password), host, dbName)

	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	brc := bedrockruntime.NewFromConfig(cfg)

	embeddingModel, err := bedrock.NewBedrock(bedrock.WithClient(brc), bedrock.WithModel(bedrock.ModelTitanEmbedG1))

	if err != nil {
		log.Fatal(err)
	}
	conn, err := pgx.Connect(context.Background(), pgConnURL)

	if err != nil {
		log.Fatal(err)
	}

	store, err = pgvector.New(
		context.Background(),
		//pgvector.WithPreDeleteCollection(true),
		pgvector.WithConn(conn),
		pgvector.WithEmbedder(embeddingModel),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("vector store ready")

	llm, err = bedrock_llm.New(bedrock_llm.WithClient(brc), bedrock_llm.WithModel(modelID))
	//llm.CallbacksHandler = callbacks.LogHandler{}

	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	load()
	query()
}

const defaultSourceURL = "https://docs.aws.amazon.com/bedrock/latest/userguide/br-studio.html"

func load() {

	source := os.Getenv("SOURCE_URL")
	if source == "" {
		source = defaultSourceURL
	}

	fmt.Println("loading data from", source)

	docs, err := getDocs(source)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("no. of document chunks to be loaded", len(docs))

	_, err = store.AddDocuments(context.Background(), docs)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("data successfully loaded into vector store")
}

func getDocs(source string) ([]schema.Document, error) {
	resp, err := http.Get(source)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()

	docs, err := documentloaders.NewHTML(resp.Body).LoadAndSplit(context.Background(), textsplitter.NewRecursiveCharacter())

	if err != nil {
		return nil, err
	}

	return docs, nil
}

func query() {

	reader := bufio.NewReader(os.Stdin)

	numOfResults := 5

	for {
		fmt.Print("\nEnter your message: ")
		question, _ := reader.ReadString('\n')
		question = strings.TrimSpace(question)

		result, err := chains.Run(
			context.Background(),
			retrievalQAChainWithCustomPrompt(llm, vectorstores.ToRetriever(store, numOfResults)),
			question,
			chains.WithMaxTokens(maxTokenCountLimitForTitanTextPremier),
		)

		if err != nil {
			log.Fatal(err)
		}

		fmt.Println("[Model response]:", result)

	}

}

// prompt based on knowledge base
const ragPromptTemplateString = `
A chat between a curious User and an artificial intelligence Bot. The Bot gives helpful, detailed, and polite answers to the User's questions.

In this session, the model has access to search results and a user's question, your job is to answer the user's question using only information from the search results.

Model Instructions:
- You should provide concise answer to simple questions when the answer is directly contained in search results, but when comes to yes/no question, provide some details.
- In case the question requires multi-hop reasoning, you should find relevant information from search results and summarize the answer based on relevant information with logical reasoning.
- If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question, and if search results are completely irrelevant, say that you could not find an exact answer, then summarize search results.
- DO NOT USE INFORMATION THAT IS NOT IN SEARCH RESULTS!

User: {{.question}}
Resource: Search Results: {{.context}} Bot:`

func retrievalQAChainWithCustomPrompt(llm *bedrock_llm.LLM, retriever vectorstores.Retriever) chains.Chain {

	ragPromptTemplate := prompts.NewPromptTemplate(
		ragPromptTemplateString,
		[]string{"context", "question"},
	)

	qaPromptSelector := chains.ConditionalPromptSelector{
		DefaultPrompt: ragPromptTemplate,
	}

	prompt := qaPromptSelector.GetPrompt(llm)

	llmChain := chains.NewLLMChain(llm, prompt)
	stuffDocsChain := chains.NewStuffDocuments(llmChain)

	return chains.NewRetrievalQA(
		stuffDocsChain,
		retriever,
	)
}
