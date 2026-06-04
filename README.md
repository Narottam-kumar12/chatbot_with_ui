# AI Chatbot with LangGraph & Gemini

A production-ready conversational AI chatbot built using **LangGraph**, **Google Gemini**, and **LangChain**, featuring persistent conversation memory and state management.

## Overview

This project demonstrates how to build a stateful AI chatbot using LangGraph's graph-based workflow architecture. The chatbot maintains conversation context across interactions and leverages Google's Gemini models for natural language understanding and generation.

### Key Features

* Stateful conversations using LangGraph
* Google Gemini integration
* Conversation memory with checkpointing
* Modular graph-based architecture
* Environment-based configuration
* Extensible workflow design
* Clean separation of state and business logic

---

## Architecture

```text
┌─────────────┐
│ User Input  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Chat State  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Chat Node   │
│ (Gemini)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Response    │
└─────────────┘
```

### Workflow

1. User sends a message.
2. Message is added to the graph state.
3. Gemini processes the conversation history.
4. Response is generated.
5. Checkpointer stores the updated conversation state.
6. Response is returned to the user.

---

## Tech Stack

* Python 3.10+
* LangGraph
* LangChain
* Google Gemini API
* Python Dotenv

---

## Project Structure

```text
.
├── chatbot.py
├── .env
├── requirements.txt
├── README.md
└── venv/
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd chatbot-project
```

### Create Virtual Environment

macOS/Linux

```bash
python3 -m venv myenv
source myenv/bin/activate
```

Windows

```bash
python -m venv myenv
myenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key
```

---

## Running the Application

```bash
python chatbot.py
```

---

## Example Usage

```python
from langchain_core.messages import HumanMessage

config = {
    "configurable": {
        "thread_id": "1"
    }
}

response = chatbot.invoke(
    {
        "messages": [
            HumanMessage(content="Hello")
        ]
    },
    config=config
)

print(response["messages"][-1].content)
```

---

## Core Components

### Chat State

Maintains the conversation history.

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### Chat Node

Processes messages through Gemini and returns a response.

```python
def chat_node(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### Checkpointer

Stores conversation state across graph executions.

```python
checkpointer = InMemorySaver()
```

---

## Future Improvements

* Streamlit UI
* Multi-user chat sessions
* Persistent database-backed memory
* RAG (Retrieval-Augmented Generation)
* Tool Calling
* Function Calling
* Agentic Workflows
* Conversation Analytics
* Authentication & Authorization
* Docker Deployment

---

## Performance Considerations

* Lightweight graph execution
* Memory-efficient state management
* Easy migration to persistent storage backends
* Scalable architecture for production environments

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## License

This project is licensed under the MIT License.

---

## Author

Narottam Kumar

B.Tech, Computer Science & Engineering

Madan Mohan Malaviya University of Technology
