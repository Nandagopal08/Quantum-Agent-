import { act, renderHook } from "@testing-library/react";
import { useAction } from "../hooks/useApi";

test("useAction calls the latest function (forms submit current values, not the first render's)", async () => {
  const seen = [];
  const { result, rerender } = renderHook(({ value }) => useAction(async () => { seen.push(value); return value; }), {
    initialProps: { value: "initial" },
  });
  rerender({ value: "edited" });
  await act(async () => { await result.current.run(); });
  expect(seen).toEqual(["edited"]);
  expect(result.current.result).toBe("edited");
});

test("useAction captures errors instead of throwing", async () => {
  const { result } = renderHook(() => useAction(async () => { throw new Error("boom"); }));
  await act(async () => { await result.current.run(); });
  expect(result.current.error).toBe("boom");
  expect(result.current.loading).toBe(false);
});
