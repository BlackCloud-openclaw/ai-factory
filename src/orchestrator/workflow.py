from langgraph.graph import StateGraph, END


def create_workflow():
    workflow = StateGraph(dict)
    
    workflow.add_node("research", lambda s: {"state": "RESEARCH"})
    workflow.add_node("code", lambda s: {"state": "CODE"})
    workflow.add_node("review", lambda s: {"state": "REVIEW"})
    
    workflow.set_entry_point("research")
    workflow.add_edge("research", "code")
    workflow.add_edge("code", "review")
    workflow.add_edge("review", END)
    
    return workflow.compile()
