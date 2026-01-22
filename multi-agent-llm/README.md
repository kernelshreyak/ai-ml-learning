## Multi-Agent LLM Examples (Python)

This folder includes standalone Python examples focused on multi-agent patterns.

- **negotiation_agents.py**: Simulates a buyer–seller price negotiation with structured outputs, market/product context, and personality-tuned prompts. The seller has a list price and floor, the buyer pushes for discounts, and either side can end the deal (accept/agree or buyer walks away). Runs for a capped number of cycles with streamed, tagged dialogue and a final price (or no-deal) summary. Useful for studying strategic turn-taking, pricing anchors, and termination rules.
- **two_agent_student_teacher.py**: Demonstrates a two-agent student/teacher setup where one agent asks questions and the other guides or corrects. Useful for studying instruction-following, feedback loops, and role conditioning.
