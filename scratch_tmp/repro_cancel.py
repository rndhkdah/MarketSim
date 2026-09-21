from pathlib import Path
from fastapi.testclient import TestClient
from marketsim.api.rest import create_app, world_admin_token
from marketsim.api.sessions import WorldManager

def auth(t): return {"Authorization": f"Bearer {t}"}

mgr = WorldManager()
app = create_app(manager=mgr, config_dir=Path("config"))
with TestClient(app) as c:
    r = c.post("/v1/worlds", json={"seed": 7, "mode": "professional", "run_mode": "lockstep"})
    st = r.json(); wid = st["world_id"]
    admin = world_admin_token(wid, st["seed"])
    a = c.post(f"/v1/worlds/{wid}/agents", json={"agent_id": "alice", "role": "agent"}, headers=auth(admin)).json()
    b = c.post(f"/v1/worlds/{wid}/agents", json={"agent_id": "bob", "role": "agent"}, headers=auth(admin)).json()
    o = c.post(f"/v1/worlds/{wid}/orders", json={"symbol": "EQ.MFG", "side": "buy", "qty": 1, "order_type": "market"}, headers=auth(a["token"]))
    print("alice submit:", o.status_code, o.json())
    oid = o.json()["order_id"]
    lb = c.get(f"/v1/worlds/{wid}/orders", headers=auth(b["token"]))
    print("bob list sees:", lb.status_code, lb.json())
    d = c.delete(f"/v1/worlds/{wid}/orders/{oid}", headers=auth(b["token"]))
    print("bob cancels alice order:", d.status_code, d.json())
    print("alice open_orders counter:", mgr._worlds[wid].agents["alice"].open_orders)
