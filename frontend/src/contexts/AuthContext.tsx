"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { onAuthStateChanged, User, signOut as firebaseSignOut } from "firebase/auth";
import { auth } from "../lib/firebase";

export interface ClinicianUser {
  uid: string;
  email: string | null;
  displayName?: string | null;
  isDemo?: boolean;
}

interface AuthContextType {
  user: ClinicianUser | null;
  loading: boolean;
  token: string | null;
  loginDemo: (email?: string) => void;
  getFreshToken: () => Promise<string | null>;
  logout: () => Promise<void>;
}

const STORAGE_KEY = "avertcare_clinician_session";

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  token: null,
  loginDemo: () => {},
  getFreshToken: async () => null,
  logout: async () => {},
});

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<ClinicianUser | null>(null);
  const [rawFirebaseUser, setRawFirebaseUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState<string | null>(null);

  // Check saved persistent session on initial mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        setUser(parsed);
        setToken("dev-clinician-token");
      }
    } catch {
      // Ignore localStorage errors
    }

    // Listen to Firebase auth state
    const unsubscribe = onAuthStateChanged(auth, async (fbUser) => {
      setRawFirebaseUser(fbUser);
      if (fbUser) {
        try {
          const jwt = await fbUser.getIdToken();
          setToken(jwt);
          const clinicianUser: ClinicianUser = {
            uid: fbUser.uid,
            email: fbUser.email,
            displayName: fbUser.displayName || "Dr. Clinician",
            isDemo: false,
          };
          setUser(clinicianUser);
          localStorage.setItem(STORAGE_KEY, JSON.stringify(clinicianUser));
        } catch {
          // If token fetch fails, keep current session
        }
      } else {
        // Only clear if no saved persistent session
        const saved = localStorage.getItem(STORAGE_KEY);
        if (!saved) {
          setUser(null);
          setToken(null);
        }
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const getFreshToken = async (): Promise<string | null> => {
    if (rawFirebaseUser) {
      try {
        // getIdToken(false) automatically refreshes if expired using the refresh token
        const freshJwt = await rawFirebaseUser.getIdToken(false);
        setToken(freshJwt);
        return freshJwt;
      } catch (err) {
        console.warn("Silent token refresh failed, falling back to cached token", err);
      }
    }
    // Check persistent session token
    if (user) {
      return token || "dev-clinician-token";
    }
    return null;
  };

  const loginDemo = (email: string = "doctor@avertcare.local") => {
    const demoUser: ClinicianUser = {
      uid: "clinician-persisted-session",
      email,
      displayName: "Dr. Clinician (Persistent Session)",
      isDemo: true,
    };
    setUser(demoUser);
    setToken("dev-clinician-token");
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(demoUser));
    } catch {
      // Ignore localStorage errors
    }
  };

  const logout = async () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore
    }
    try {
      await firebaseSignOut(auth);
    } catch {
      // Ignore
    }
    setUser(null);
    setRawFirebaseUser(null);
    setToken(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, token, loginDemo, getFreshToken, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
