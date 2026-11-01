from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_runs_create_and_return_result(monkeypatch):
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
    def fake_run_graph(field, *, events):
        return {'field': field.model_dump(), 'events': events}

    monkeypatch.setattr('api.main.run_graph', fake_run_graph)
    response = client.post('/runs', json=payload)
    assert response.status_code == 200
    run_id = response.json()['run_id']
    events = client.get(f'/runs/{run_id}/events')
    assert events.status_code == 200
    body = events.text
    assert 'event: update' in body
    assert 'loading weather and satellite observations' in body
    assert 'data:' in body
    final = client.get(f'/runs/{run_id}')
    assert final.status_code == 200
    assert 'field' in final.json()


def test_bad_polygon_is_rejected_with_human_readable_422():
    payload = {
        'polygon': {'type': 'LineString', 'coordinates': [[78.0, 17.0], [78.1, 17.0]]},
        'as_of': '2025-08-20',
    }
    response = client.post('/runs', json=payload)
    assert response.status_code == 422
    detail = response.json()['detail']
    assert any('polygon' in str(item).lower() for item in detail)


def test_unknown_crop_is_rejected_before_background_run():
    response = client.post(
        '/runs',
        json={
            'polygon': {
                'type': 'Polygon',
                'coordinates': [[[78.0, 17.0], [78.1, 17.0], [78.1, 17.1], [78.0, 17.1], [78.0, 17.0]]],
            },
            'as_of': '2025-08-20',
            'crop': 'unknown',
        },
    )
    assert response.status_code == 422
    assert any('crop' in str(item).lower() for item in response.json()['detail'])


def test_invalid_polygon_coordinates_are_rejected_before_background_run():
    response = client.post(
        '/runs',
        json={
            'polygon': {
                'type': 'Polygon',
                'coordinates': [[['not-a-number', 17.0], [78.1, 17.0], [78.1, 17.1]]],
            },
            'as_of': '2025-08-20',
        },
    )
    assert response.status_code == 422
    assert any('polygon is invalid' in str(item).lower() for item in response.json()['detail'])


def test_pipeline_error_is_returned_as_error_not_processing():
    payload = {
        'polygon': {
            'type': 'Polygon',
            'coordinates': [[[78.0, 17.0], [78.1, 17.0], [78.1, 17.1], [78.0, 17.1], [78.0, 17.0]]],
        },
        'as_of': '2025-08-20',
    }

    with patch('api.main.run_graph', side_effect=RuntimeError('test-only pipeline failure')):
        response = client.post('/runs', json=payload)
        assert response.status_code == 200
        run_id = response.json()['run_id']
        events = client.get(f'/runs/{run_id}/events')
        result = client.get(f'/runs/{run_id}')

    assert '"node": "respond", "status": "error"' in events.text
    assert result.json() == {
        'run_id': run_id,
        'status': 'error',
        'error': 'test-only pipeline failure',
    }
