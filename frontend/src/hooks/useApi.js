import { useState, useEffect, useCallback } from 'react';
import { apiGet, apiPost } from '../api';

export function useApi(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(!!url);
  const [error, setError] = useState(null);

  const refetch = useCallback(() => {
    if (!url) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    apiGet(url)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [url]);

  useEffect(() => {
    if (!url) {
      setData(null);
      setLoading(false);
      return;
    }
    refetch();
  }, [refetch, url]);

  return { data, loading, error, refetch };
}

export function useMutation(method = 'POST') {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const mutate = useCallback(
    async (url, body) => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiPost(url, body);
        return res;
      } catch (e) {
        setError(e.message);
        throw e;
      } finally {
        setLoading(false);
      }
    },
    [method]
  );

  return { mutate, loading, error };
}
