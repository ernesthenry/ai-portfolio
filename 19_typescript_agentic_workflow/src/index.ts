import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import * as dotenv from "dotenv";

dotenv.config();

// 1. DATA MODELING WITH ZOD (Equivalent to Pydantic)
// "Strong understanding of data modeling" -> JD Requirement
const AnalystSchema = z.object({
  sentiment: z.enum(["Positive", "Neutral", "Negative"]).describe("The overall mood of the text"),
  key_entities: z.array(z.string()).describe("List of people or organizations mentioned"),
  action_items: z.array(z.string()).describe("List of implied tasks or to-dos"),
  risk_score: z.number().min(0).max(10).describe("Risk level from 0 to 10")
});

// 2. INITIALIZE MODEL (OpenAI)
const model = new ChatOpenAI({
  temperature: 0,
  modelName: "gpt-3.5-turbo",
  // In LangChain.js, structured output is handled via 'withStructuredOutput'
});

async function runAgent() {
  console.log("🚀 Starting TypeScript AI Agent...");

  const textToAnalyze = `
    Subject: Project O Update
    From: CEO (AI)
    
    Team, we are moving too slow. The sovereign ecosystem needs to launch by Q3.
    I need Engineering to prioritize the Vector Store integration immediately.
    Marketing needs to draft the whitepaper.
    There is a high risk of competitor entry if we delay.
  `;

  // 3. DEFINE CHAIN
  // Combining Prompt + Model + Schema Validation
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are an expert AI Analyst for OFoundation. Extract structured insights from internal comms."],
    ["human", "{text}"],
  ]);

  // Bind the Zod schema to the model (Function Calling under the hood)
  const structuredLlm = model.withStructuredOutput(AnalystSchema);
  const chain = prompt.pipe(structuredLlm);

  console.log(`\nAnalyzing Text:\n"${textToAnalyze.trim().substring(0, 50)}..."\n`);

  try {
    // 4. EXECUTE
    const result = await chain.invoke({
      text: textToAnalyze,
    });

    console.log("✅ API Integration Successful. Structured Output:");
    console.log(JSON.stringify(result, null, 2));

    // 5. BUSINESS LOGIC (TypeScript)
    // "Implement iterate on user-facing AI features"
    if (result.risk_score > 7) {
        console.log("\n⚠️ HIGH RISK ALERT: Triggering escalation protocol...");
    }

  } catch (error) {
    console.error("❌ Error running agent:", error);
  }
}

// Run if called directly
if (require.main === module) {
  runAgent();
}
