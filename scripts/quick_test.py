import json, os, requests, time, pprint

BASE = os.getenv("BASE_URL","http://localhost:8009")
pp = pprint.PrettyPrinter(indent=2)

def call(scene, goal):
    payload = {
        "model": "mindgrid-robot-llm-starter",
        "messages": [
            {"role":"user","content": json.dumps({"scene": scene, "goal": goal})}
        ],
        "tools": [{
            "type":"function",
            "function":{"name":"robot.plan","parameters":{"type":"object"}}
        }]
    }
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload)
    r.raise_for_status()
    data = r.json()
    tool = data["choices"][0]["message"]["tool_calls"][0]
    args = json.loads(tool["function"]["arguments"])
    return args

def main():
    print("Health:", requests.get(f"{BASE}/healthz").json())

    # Kitchen test
    goal1 = "Place mug_17 on counter_A"
    print("\n=== Kitchen test ===")
    plan1 = call(json.load(open("samples/scene_kitchen.json")), goal1)
    pp.pprint(plan1)

    # Drawer test
    goal2 = "Open the drawer"
    print("\n=== Drawer test ===")
    plan2 = call(json.load(open("samples/scene_drawer.json")), goal2)
    pp.pprint(plan2)

if __name__ == "__main__":
    main()
