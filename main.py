import os
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

# 🌱 Load .env
load_dotenv()


def setup_agents():
    external_client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client,
    )

    config = RunConfig(model=model, model_provider=external_client)

    # 📖 Narrator Agent: Story flow
    narrator_agent = Agent(
        name="narrator_agent",
        instructions="Narrate the fantasy adventure based on player choices.",
        handoff_description="Tells the story and responds to choices.",
        model=model,
    )

    # 🧟 Monster Agent: Handles combat and challenges
    monster_agent = Agent(
        name="monster_agent",
        instructions="Simulate combat using roll_dice() and generate_event().",
        handoff_description="Manages monster battles.",
        model=model,
        tools=[
            {
                "tool_name": "roll_dice",
                "tool_description": "Rolls a 20-sided dice.",
                "tool": lambda _: f"🎲 Dice Roll: {__import__('random').randint(1, 20)}"
            },
            {
                "tool_name": "generate_event",
                "tool_description": "Creates a random combat event.",
                "tool": lambda _: "🧟 You encountered a wild goblin ambush!"
            }
        ]
    )

    # 🎁 Item Agent: Rewards and inventory
    item_agent = Agent(
        name="item_agent",
        instructions="Manage player's inventory and rewards.",
        handoff_description="Handles inventory, items, and magical loot.",
        model=model,
    )

    # 🎮 Game Master Agent: Delegates based on context
    game_master_agent = Agent(
        name="game_master_agent",
        instructions=(
            "You are the game master for a fantasy adventure game. Based on the player's input,"
            "choose the appropriate agent (narrator, combat, item). Never respond directly yourself."
        ),
        tools=[
            narrator_agent.as_tool("advance_story", "Narrates the adventure."),
            monster_agent.as_tool("combat_phase", "Handles enemy encounters."),
            item_agent.as_tool("manage_inventory",
                               "Manages items and rewards."),
        ],
        model=model,
    )

    return game_master_agent, config

# 🧙 Chainlit start


@cl.on_chat_start
async def start():
    agent, config = setup_agents()
    cl.user_session.set("agent", agent)
    cl.user_session.set("config", config)
    await cl.Message("🧙 Welcome, adventurer! Your fantasy quest begins now. Type anything to start...").send()

# 🎲 User input handler


@cl.on_message
async def handle(message: cl.Message):
    thinking = cl.Message("🧠 Thinking...")
    await thinking.send()

    agent = cl.user_session.get("agent")
    config = cl.user_session.get("config")

    result = await Runner.run(agent, [{"role": "user", "content": message.content}], run_config=config)

    thinking.content = result.final_output
    await thinking.update()
