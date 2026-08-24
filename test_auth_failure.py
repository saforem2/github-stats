#!/usr/bin/python3
"""
Regression tests for silent authentication failures.

Background: GitHub returns HTTP 401/403 with a *valid JSON* body
(e.g. {"message": "Bad credentials"}). The original Queries.query() only
treated status >= 500 as an error, so an auth failure was returned to the
caller as if it were a successful result. Stats.get_stats() then read
.get("data", {}) off that error body, found nothing, and silently produced
a badge reading "No Name" with every statistic zeroed -- while the workflow
exited 0 and committed the empty SVGs over the good ones.
"""

import asyncio
import unittest
from unittest.mock import patch

from github_stats import Queries, Stats


class FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status, payload):
        self.status = status
        self._payload = payload
        self.request_info = None
        self.history = ()

    async def json(self):
        return self._payload

    def release(self):
        pass


class FakeSession:
    """aiohttp.ClientSession stand-in returning a canned response."""

    def __init__(self, status, payload):
        self.status = status
        self.payload = payload
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse(self.status, self.payload)

    async def get(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse(self.status, self.payload)


BAD_CREDENTIALS = {
    "message": "Bad credentials",
    "documentation_url": "https://docs.github.com/rest",
    "status": "401",
}


def run(coro):
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


class TestGraphQLAuthFailure(unittest.TestCase):
    def test_401_raises_instead_of_returning_error_body(self):
        """A 401 must raise, not be handed back as a successful result."""
        session = FakeSession(401, BAD_CREDENTIALS)
        q = Queries("saforem2", "bad-token", session)

        with patch("requests.post", side_effect=RuntimeError("no network")):
            with self.assertRaises(RuntimeError) as ctx:
                run(q.query("query { viewer { login } }"))

        self.assertIn("rejected the access token", str(ctx.exception).lower())

    def test_401_is_not_retried(self):
        """Auth failures are terminal; retrying 10x cannot fix a bad token."""
        session = FakeSession(401, BAD_CREDENTIALS)
        q = Queries("saforem2", "bad-token", session)

        with patch("requests.post", side_effect=RuntimeError("no network")):
            with self.assertRaises(RuntimeError):
                run(q.query("query { viewer { login } }"))

        self.assertEqual(
            session.calls, 1, "a 401 should fail fast, not burn retry attempts"
        )

    def test_403_also_raises(self):
        """403 (bad scopes / blocked token) is equally terminal."""
        session = FakeSession(403, {"message": "Forbidden"})
        q = Queries("saforem2", "bad-token", session)

        with patch("requests.post", side_effect=RuntimeError("no network")):
            with self.assertRaises(RuntimeError):
                run(q.query("query { viewer { login } }"))

    def test_graphql_errors_block_are_surfaced(self):
        """
        GraphQL can return HTTP 200 with a top-level "errors" block and no
        usable "data". That is a failure too, and must not silently zero out.
        """
        session = FakeSession(
            200, {"errors": [{"type": "FORBIDDEN", "message": "Bad credentials"}]}
        )
        q = Queries("saforem2", "bad-token", session)

        with patch("requests.post", side_effect=RuntimeError("no network")):
            with self.assertRaises(RuntimeError):
                run(q.query("query { viewer { login } }"))

    def test_successful_query_still_returns_data(self):
        """The happy path must be untouched by the new error handling."""
        payload = {"data": {"viewer": {"login": "saforem2", "name": "Sam Foreman"}}}
        session = FakeSession(200, payload)
        q = Queries("saforem2", "good-token", session)

        result = run(q.query("query { viewer { login } }"))
        self.assertEqual(result, payload)
        self.assertEqual(session.calls, 1)


class TestRestAuthFailure(unittest.TestCase):
    def test_rest_401_raises(self):
        session = FakeSession(401, BAD_CREDENTIALS)
        q = Queries("saforem2", "bad-token", session)

        with patch("requests.get", side_effect=RuntimeError("no network")):
            with self.assertRaises(RuntimeError) as ctx:
                run(q.query_rest("/repos/saforem2/github-stats/traffic/views"))

        self.assertIn("rejected the access token", str(ctx.exception).lower())

    def test_rest_403_is_skipped_not_fatal(self):
        """
        403 on a REST path is normally per-repo ("Must have push access to
        repository") for a repo the user contributed to but does not own.
        That must skip the repo, not abort the whole run -- otherwise a
        single inaccessible repo produces no badge at all.
        """
        session = FakeSession(403, {"message": "Must have push access to repository"})
        q = Queries("saforem2", "good-token", session)

        result = run(q.query_rest("/repos/pytorch/pytorch/traffic/views"))
        self.assertEqual(result, dict())

    def test_rest_204_still_returns_empty_dict(self):
        """
        204 No Content is a legitimate empty response (repos with no
        contributor stats) and must keep working, not become an error.
        """
        session = FakeSession(204, None)
        q = Queries("saforem2", "good-token", session)

        result = run(q.query_rest("/repos/saforem2/x/stats/contributors"))
        self.assertEqual(result, dict())


class TestStatsDoesNotDegradeSilently(unittest.TestCase):
    def test_get_stats_propagates_auth_error(self):
        """
        The end-to-end symptom: a bad token must not yield a 'No Name'
        badge with zeroed stats. get_stats() has to blow up instead.
        """
        session = FakeSession(401, BAD_CREDENTIALS)
        s = Stats("saforem2", "bad-token", session)

        with patch("requests.post", side_effect=RuntimeError("no network")):
            with self.assertRaises(RuntimeError):
                run(s.get_stats())


if __name__ == "__main__":
    unittest.main(verbosity=2)
