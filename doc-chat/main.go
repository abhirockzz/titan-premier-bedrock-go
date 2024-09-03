package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	bedrock_llm "github.com/tmc/langchaingo/llms/bedrock"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
)

var llm *bedrock_llm.LLM

const modelID = "amazon.titan-text-premier-v1:0"

// const modelID = bedrock_llm.ModelAnthropicClaudeV3Sonnet
const maxTokenCountLimitForTitanTextPremier = 3072

func init() {

	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	brc := bedrockruntime.NewFromConfig(cfg)

	llm, err = bedrock_llm.New(bedrock_llm.WithClient(brc), bedrock_llm.WithModel(modelID))
	//llm.CallbacksHandler = callbacks.LogHandler{}

	if err != nil {
		log.Fatal(err)
	}
}

const defaultSourceURL = "https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html"

func main() {

	reader := bufio.NewReader(os.Stdin)

	link := os.Getenv("SOURCE_URL")
	if link == "" {
		link = defaultSourceURL
	}

	docs := getDocs(link)

	for {
		fmt.Print("\nEnter your message: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		answer, err := chains.Call(
			context.Background(),
			docChainWithCustomPrompt(llm),
			map[string]any{
				"input_documents": docs,
				"question":        input,
			},
			chains.WithMaxTokens(maxTokenCountLimitForTitanTextPremier))

		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("[Response from model]:", answer["text"])
	}

}

// prompt based on - https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-templates-and-examples.html#qa-with-context
const promptTemplateString = "{{.context}}\nBased on the information above, {{.question}}"

func docChainWithCustomPrompt(llm *bedrock_llm.LLM) chains.Chain {

	ragPromptTemplate := prompts.NewPromptTemplate(
		promptTemplateString,
		[]string{"context", "question"},
	)

	qaPromptSelector := chains.ConditionalPromptSelector{
		DefaultPrompt: ragPromptTemplate,
	}

	prompt := qaPromptSelector.GetPrompt(llm)

	llmChain := chains.NewLLMChain(llm, prompt)
	return chains.NewStuffDocuments(llmChain)

}

func getDocs(link string) []schema.Document {

	resp, err := http.Get(link)
	if err != nil {
		log.Fatal(err)
	}

	defer resp.Body.Close()

	docs, err := documentloaders.NewHTML(resp.Body).Load(context.Background())

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("loaded content from", link)

	return docs
}
