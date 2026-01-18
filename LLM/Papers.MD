# TTT-E2E
Instead of relying on direct reference to all past tokens via the KV-cache in the original Transformer, TTT-E2E updates the model itself so that contextual information is compressed into the model parameters in a goal-conditioned manner, driven by the next-token prediction loss.

This approach does not preserve exact token-level recall and is therefore weaker at tasks like needle-in-a-haystack retrieval. However, it exhibits scaling behavior comparable to full attention on tasks where reasoning and logical consistency are emphasized, by progressively consolidating relevant information into the model state.

If exact recall is required, this limitation can be mitigated by augmenting the model with an additional hard lookup or external memory mechanism.


# RLMs (Recursive Language Models)

Instead of ingesting long contexts directly, RLMs treat the prompt as a variable in an external, programmatic environment (e.g., a REPL), rather than as neural input.
The LLM explores, decomposes, and manipulates the prompt by writing code, and solves problems by recursively calling itself or sub-LMs when needed.
During this process, the model retains primary control over context access, strategy selection, and verification within the environment.
Through selective, symbolic interaction with the prompt, RLMs maintain strong performance on contexts that exceed the base model’s context window by orders of magnitude.


# MCP (Model Context Protocol)
People want to expand the potential of LLM that's so powerful that makes just directly using its parameterized knowledge less desired. So there's a need for tool calling, which again raised an issue related with consistent formatting for compatibility and reusability. This is contextual background of MCP. 

Then MCP make consistent the information it has, it wants to get, and return in terms of FORMAT. 
It will have 
 - describe method to explain what a function can do 
 - call method where
	- will get arguments as input parameter. So internally at each method, function can access to the arguments 
       - return dictionary
## Two level understanding
### Super High Level
```
class MCPTool:
    def describe(self) -> dict:
        """What this tool can do"""
    
    def call(self, arguments: dict) -> dict:
        """Standardized Calling"""
```

### Little bit lower
```
# What you would get as a description of a tool 
{
  "name": "get_weather",
  "description": "Get current weather for a given city",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": { "type": "string" }
    },
    "required": ["city"]
  }
}

# Tool calling detail
def call(arguments: dict) -> dict:
    city = arguments["city"]
    return {
        "temperature": 23,
        "condition": "sunny"
    }
```

## Large Components of MCP
Client, Server, and Tools

Tools:
- Business logic
- Perform actual operations and side effects

Server:
- Hosts and exposes tools (and resources/prompts)
- Acts as the execution and security boundary
- Makes tools accessible via MCP protocol

Client:
- Uses an LLM to determine whether external tools are needed
- Decides which tool to call and with what arguments
- Issues tool invocation requests via MCP protocol
