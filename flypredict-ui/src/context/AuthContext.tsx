import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

type AuthContextValue = {
  isLoggedIn: boolean;
  login: (user: string) => void;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const AUTH_KEY = "flypredict_auth";
const USER_KEY = "flypredict_user";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isLoggedIn, setIsLoggedIn] = useState(() => localStorage.getItem(AUTH_KEY) === "1");

  useEffect(() => {
    const syncAuth = () => {
      setIsLoggedIn(localStorage.getItem(AUTH_KEY) === "1");
    };

    window.addEventListener("storage", syncAuth);
    return () => window.removeEventListener("storage", syncAuth);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      isLoggedIn,
      login: (user: string) => {
        localStorage.setItem(AUTH_KEY, "1");
        localStorage.setItem(USER_KEY, user);
        sessionStorage.setItem(AUTH_KEY, "1");
        sessionStorage.setItem(USER_KEY, user);
        setIsLoggedIn(true);
      },
      logout: () => {
        localStorage.removeItem(AUTH_KEY);
        localStorage.removeItem(USER_KEY);
        sessionStorage.removeItem(AUTH_KEY);
        sessionStorage.removeItem(USER_KEY);
        setIsLoggedIn(false);
      },
    }),
    [isLoggedIn],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
