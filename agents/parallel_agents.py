import time
import asyncio
import matplotlib.pyplot as plt
import nest_asyncio
from agents import Agent,Runner,ModelSettings
import os
from dotenv import load_dotenv


load_dotenv('./.env')

GPT_MODEL ='gpt-5'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")

features_agent =Agent(
    name='FeaturesAgent',
    instructions= 'Extract the key product features from the review.'
)

pros_cons_agent =Agent(
    name ='ProsConsAgent',
    instructions='List the pros and cons mentioned in the review.'
)
sentiment_agent = Agent(
    name="SentimentAgent",
    instructions="Summarize the overall user sentiment from the review."
)
recommend_agent = Agent(
    name="RecommendAgent",
    instructions="State whether you would recommend this product and why."
)
parallel_agents=[
    features_agent,
    pros_cons_agent,
    sentiment_agent,
    recommend_agent
]
meta_agent_parrel_tools=Agent(
    name="MetaAgent",
    instructions="You are given multiple summaries labeled with Features, ProsCons, Sentiment, and a Recommendation."

)

meta_agent =Agent(
    name='MetaAgent',
    instructions='You are given multiple summaries labeled with Features,ProsCOns,Sentiment,and a Recommendation.'
    " Combine them into a concise executive summary of the product review with a 1-5 star rating for each summary area.",
    
    model_settings=ModelSettings(
       parallel_tool_calls=True
   ),
   tools= [
       features_agent.as_tool(
            tool_name="features",
            tool_description="Extract the key product features from the review.",   
       ),
        pros_cons_agent.as_tool(
            tool_name="pros_cons",
            tool_description="List the pros and cons mentioned in the review.",
        ),
        sentiment_agent.as_tool(
            tool_name="sentiment",
            tool_description="Summarize the overall user sentiment from the review.",
        ),
        recommend_agent.as_tool(
            tool_name="recommend",
            tool_description="State whether you would recommend this product and why.",
        ),
   ]
)
starts ,ends =[],[]
async def run_agent(agent,review_text:str):
    agent_name = agent.name
    start =time.time()
    starts.append((agent_name,start))

    result =await Runner.run(agent,review_text)
    
    end =time.time()
    ends.append((agent_name,end))
    return result

async def run_agents(review_text:str):
    """asyncio.gather(*coroutines)
用于 并发执行多个 awaitable 对象（协程），并在全部完成后一次性返回结果。"""
    responses =await asyncio.gather(
        *(run_agent(agent,review_text) for agent in parallel_agents)
    )

    labeled_summaries =[
        f"### {resp.last_agent.name}\n {resp.final_output}"
        for resp in responses
    ]

    collected_summaries ="\n".join(labeled_summaries)
    final_summary =await run_agent(meta_agent,collected_summaries)

    print('Final summary:',final_summary.final_output)

    return

def plot_timeline(starts,ends):
    base = min(t for _, t in starts)
    labels = [n for n, _ in starts]
    start_offsets = [t - base for _, t in starts]
    lengths = [ends[i][1] - starts[i][1] for i in range(len(starts))]

    plt.figure(figsize=(8, 3))
    plt.barh(labels, lengths, left=start_offsets)
    plt.xlabel("Seconds since kickoff")
    plt.title("Agent Execution Timeline")
    #plt.show()
    plt.savefig("agent_timeline.png", bbox_inches="tight")
    plt.close()

async def main():
    review_text = """
    I recently upgraded to the AuroraSound X2 wireless noise-cancelling headphones, and after two weeks of daily use I have quite a bit to share. First off, the design feels premium without being flashy: the matte‐finish ear cups are softly padded and rotate smoothly for storage, while the headband’s memory‐foam cushion barely presses on my temples even after marathon work calls. Connectivity is seamless—pairing with my laptop and phone took under five seconds each time, and the Bluetooth 5.2 link held rock-solid through walls and down the hallway.

    The noise-cancelling performance is genuinely impressive. In a busy café with music and chatter swirling around, flipping on ANC immediately quiets low-level ambient hums, and it even attenuates sudden noises—like the barista’s milk frother—without sounding distorted. The “Transparency” mode is equally well‐tuned: voices come through clearly, but the world outside isn’t overwhelmingly loud. Audio quality in standard mode is rich and balanced, with tight bass, clear mids, and a hint of sparkle in the highs. There’s also a dedicated EQ app, where you can toggle between “Podcast,” “Bass Boost,” and “Concert Hall” presets or craft your own curve.

    On the control front, intuitive touch panels let you play/pause, skip tracks, and adjust volume with a simple swipe or tap. One neat trick: holding down on the right ear cup invokes your phone’s voice assistant. Battery life lives up to the hype, too—over 30 hours with ANC on, and the quick‐charge feature delivers 2 hours of playtime from just a 10-minute top-up.

    That said, it isn’t perfect. For one, the carrying case is a bit bulky, so it doesn’t slip easily into a slim bag. And while the touch interface is mostly reliable, I occasionally trigger a pause when trying to adjust the cup position. The headphones also come in only two colorways—black or white—which feels limiting given the premium price point.
    """
    # asyncio.run(run_agents(review_text))


    result =await run_agent(meta_agent_parrel_tools,review_text)
    print('Final summary:', result.final_output)
    plot_timeline(starts, ends)


if __name__ == "__main__":
    asyncio.run(main())
