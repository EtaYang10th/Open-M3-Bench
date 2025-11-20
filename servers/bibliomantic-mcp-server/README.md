# Bibliomantic MCP Server

A Model Context Protocol server that integrates I Ching divination with AI responses, exploring the bibliomantic approach described in Philip K. Dick's *The Man in the High Castle*.

## Purpose

This server demonstrates how traditional wisdom systems can be thoughtfully integrated with modern AI through MCP. It provides:

- **Educational exploration** of ancient Chinese philosophy and the I Ching
- **Literary context** from Philip K. Dick's influential science fiction work
- **Philosophical reflection tools** for creative thinking and perspective-taking
- **Technical demonstration** of FastMCP capabilities with tools, resources, and prompts

## Features

- **Traditional I Ching System**: All 64 hexagrams with authentic three-coin method simulation
- **Bibliomantic Consultation**: Following Philip K. Dick's approach from "The Man in the High Castle"
- **FastMCP Implementation**: Professional MCP compliance with tools, resources, and prompts
- **Ethical Safeguards**: Clear disclaimers about entertainment/reflection purpose
- **Educational Context**: Historical and philosophical background for each consultation
- **Resource Access**: Direct hexagram database access for AI context loading

## Prerequisites

- **Python 3.10+** (required by MCP SDK)
- **MCP-compatible host** (Claude Desktop, etc.)

## Installation

### From source
```bash
git clone https://github.com/d4nshields/bibliomantic-mcp-server.git
cd bibliomantic-mcp-server
pip install -e .
```

Alternative configurations:

```json
{
  "mcpServers": {
    "bibliomantic": {
      "command": "python",
      "args": ["-m", "bibliomantic_server"]
    }
  }
}
```

## Available Tools

- **`i_ching_divination`** - Generate random hexagrams with traditional interpretations
- **`bibliomantic_consultation`** - Full bibliomantic process with query augmentation  
- **`get_hexagram_details`** - Look up specific hexagrams by number (1-64)
- **`server_statistics`** - View system information and capabilities

## Resources

- **`hexagram://{number}`** - Load individual hexagram data for AI context
- **`iching://database`** - Access complete 64-hexagram database overview

## Prompt Templates

- **`career_guidance_prompt`** - Structured prompts for career decisions
- **`creative_guidance_prompt`** - Prompts for artistic and creative projects
- **`general_guidance_prompt`** - Universal life guidance templates

## Usage Examples

### Basic Divination
Ask Claude:
> *"Can you perform an I Ching divination for philosophical guidance on my creative projects?"*

### Bibliomantic Consultation
> *"I'm facing a difficult decision about my career path. Can you consult the I Ching using the bibliomantic approach?"*

### Resource Access
> *"Load the hexagram://1 resource to understand The Creative."*

### Specific Hexagram Lookup
> *"Can you tell me about I Ching hexagram number 42?"*

## The Bibliomantic Approach

This implementation follows the divination method described in Philip K. Dick's *The Man in the High Castle*, where characters consult the I Ching for guidance on important decisions. The system:

- Uses traditional I Ching methodology with three-coin tosses
- Generates cryptographically secure randomness for authentic divination
- Integrates ancient wisdom with modern AI capabilities
- Maintains the philosophical and reflective aspects of bibliomancy
- Provides transparency in the divination process

## Ethical Framework

All responses include clear disclaimers that this is for **philosophical reflection and entertainment only**, not supernatural guidance or life advice. Users are directed to consult qualified professionals for important decisions.

The server emphasizes:
- Educational exploration of wisdom traditions
- Creative thinking and perspective-taking
- Philosophical reflection rather than prediction
- Respect for both ancient traditions and modern ethical standards

## Development

### Testing with MCP Inspector
```bash
mcp dev bibliomantic_server.py
```

### Running locally
```bash
python bibliomantic_server.py
```

## Technical Implementation

Built with:
- **FastMCP** - Official MCP Python SDK for professional compliance
- **Traditional I Ching** - Complete 64-hexagram database with authentic interpretations
- **Cryptographic randomness** - Secure three-coin simulation using Python's `secrets` module
- **Type safety** - Full type hints with automatic JSON schema generation
- **Ethical safeguards** - User-facing disclaimers in all responses

## Use Cases

- **Educational tool** for learning about I Ching philosophy and Chinese wisdom traditions
- **Creative writing aid** for generating new perspectives and inspiration
- **Philosophical reflection** tool for contemplating life decisions and changes
- **Cultural bridge** between ancient wisdom and modern AI capabilities
- **Technical demonstration** of MCP server development with cultural content
- **Literary analysis** of Philip K. Dick's themes and bibliomantic concepts

## Security Considerations

- Input validation for all user queries and parameters
- No external network requests or API dependencies
- Cryptographically secure randomness generation
- Clear ethical boundaries and user education
- No persistent data storage or user tracking

## Contributing

Contributions are welcome! Please:

1. Follow the existing code style and patterns
2. Add tests for new functionality using MCP Inspector
3. Update documentation for any changes
4. Ensure compatibility with the official MCP SDK
5. Maintain the ethical framework and educational focus

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Philip K. Dick** for the literary inspiration and bibliomantic approach
- **Ancient Chinese philosophers** for the I Ching wisdom tradition
- **Anthropic** for the Model Context Protocol and FastMCP framework
- **The MCP community** for fostering innovative AI integrations

---

*"The oracle was right. The future remains ahead."* - Philip K. Dick, The Man in the High Castle
