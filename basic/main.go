package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/bedrock"
)

const defaultRegion = "us-east-1"
const maxTokenCountLimitForTitanTextPremier = 3072
const modelID = "amazon.titan-text-premier-v1:0"

var client *bedrockruntime.Client

func init() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	client = bedrockruntime.NewFromConfig(cfg)
}

func main() {

	llm, err := bedrock.New(bedrock.WithClient(client), bedrock.WithModel(modelID))

	if err != nil {
		log.Fatal(err)
	}

	msg := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Explain AI in 100 words or less."),
			},
		},
	}

	//llm.CallbacksHandler = callbacks.LogHandler{}

	resp, err := llm.GenerateContent(context.Background(), msg, llms.WithMaxTokens(maxTokenCountLimitForTitanTextPremier))

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("response:\n", resp.Choices[0].Content)

}
