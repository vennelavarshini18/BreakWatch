from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_runs_create_and_return_result():
    payload = {
        'polygon': {
            'type': 'Polygon',
            'coordinates': [[[78.0, 17.0], [78.1, 17.0], [78.1, 17.1], [78.0, 17.1], [78.0, 17.0]]],
        },
        'as_of': '2025-08-20',
        'crop': 'cotton',
        'soil': 'black_clay',
        'irrigation_budget_mm': 5.0,
    }
    response = client.post('/runs', json=payload)
    assert response.status_code == 200
    run_id = response.json()['run_id']
    events = client.get(f'/runs/{run_id}/events')
    assert events.status_code == 200
    final = client.get(f'/runs/{run_id}')
    assert final.status_code == 200
    assert 'field' in final.json()
