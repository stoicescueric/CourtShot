import { useCallback, useEffect, useRef, useState, type DependencyList } from 'react';

export interface Resource<T> {
  data: T | undefined;
  /** The last failure. Data from an earlier success stays, so a failed poll does not blank the page. */
  error: unknown;
  reload: () => void;
}

/** Loads on mount and whenever `deps` change; answers that arrive after a newer request or unmount are dropped. */
export function useResource<T>(load: () => Promise<T>, deps: DependencyList): Resource<T> {
  const [state, setState] = useState<{ data?: T; error?: unknown }>({});
  const seq = useRef(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  const reload = useCallback(() => {
    const n = ++seq.current;
    loadRef.current().then(
      (data) => seq.current === n && setState({ data }),
      (error: unknown) => seq.current === n && setState((s) => ({ data: s.data, error })),
    );
  }, []);

  useEffect(() => {
    setState({});
    reload();
    return () => {
      seq.current += 1;
    };
  }, deps);

  return { data: state.data, error: state.error, reload };
}

/** Calls `fn` every `ms` milliseconds; `null` stops it. */
export function useInterval(fn: () => void, ms: number | null): void {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    if (ms == null) return;
    const id = setInterval(() => fnRef.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}
