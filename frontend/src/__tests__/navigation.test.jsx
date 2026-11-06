import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { AppRoutes } from "../App";
import { NAV } from "../layouts/AppLayout";

// The API is mocked to fail fast: these tests check routing and that every page renders an error state
// instead of crashing when the backend is unreachable.
vi.mock("../services/api", async (orig) => {
  const real = await orig();
  const fail = () => Promise.reject(Object.assign(new Error("Network Error"), { code: "ERR_NETWORK" }));
  return { ...real, api: new Proxy({}, { get: (_t, k) => (k.endsWith("Url") ? () => null : fail) }) };
});

test("sidebar lists all 14 modules", () => {
  expect(NAV).toHaveLength(14);
});

test.each(NAV.map((n) => [n.label, n.to]))("%s page renders at %s", async (label, path) => {
  render(<MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><AppRoutes /></MemoryRouter>);
  expect(await screen.findByRole("heading", { level: 1, name: label.replace("Digital Twin", "Digital Twin") })).toBeInTheDocument();
});
