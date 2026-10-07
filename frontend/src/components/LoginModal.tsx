"use client";

import React, { useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../lib/firebase";
import { useAuth } from "@/contexts/AuthContext";
import { Activity, Lock, Mail, ArrowRight, Loader2 } from "lucide-react";

export function LoginModal() {
  const { user, loading: authLoading } = useAuth();
  const [email, setEmail] = useState("doctor@avertcare.local");
  const [password, setPassword] = useState("pa3ssword12");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (authLoading || user) return null; // Don't show if logged in or loading

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError("");

    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err: any) {
      setError(err.message || "Failed to log in.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md">
      <div className="relative w-full max-w-md p-8 overflow-hidden bg-[#111113] border border-white/10 rounded-2xl shadow-2xl">
        {/* Glow Effects */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-24 bg-blue-500/20 blur-[60px] rounded-full pointer-events-none" />
        
        <div className="relative z-10 flex flex-col items-center">
          <div className="flex items-center justify-center w-14 h-14 mb-6 rounded-full bg-blue-500/10 border border-blue-500/20 shadow-[0_0_15px_rgba(59,130,246,0.5)]">
            <Activity className="w-7 h-7 text-blue-400" />
          </div>
          
          <h2 className="mb-2 text-2xl font-bold tracking-tight text-white">
            Clinician Portal
          </h2>
          <p className="mb-8 text-sm text-gray-400 text-center">
            Secure access to AvertCare Predictive Engine
          </p>

          <form onSubmit={handleLogin} className="w-full space-y-5">
            <div className="space-y-1">
              <label className="text-xs font-semibold tracking-wide text-gray-400 uppercase">
                Email Address
              </label>
              <div className="relative flex items-center">
                <Mail className="absolute left-3 w-5 h-5 text-gray-500" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full py-2.5 pl-10 pr-4 text-sm text-white bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all placeholder-gray-600"
                  placeholder="name@hospital.org"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold tracking-wide text-gray-400 uppercase">
                Password
              </label>
              <div className="relative flex items-center">
                <Lock className="absolute left-3 w-5 h-5 text-gray-500" />
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full py-2.5 pl-10 pr-4 text-sm text-white bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all placeholder-gray-600"
                  placeholder="••••••••"
                />
              </div>
            </div>

            {error && (
              <div className="p-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={isSubmitting}
              className="group relative flex items-center justify-center w-full py-2.5 mt-2 text-sm font-semibold text-white transition-all bg-blue-600 rounded-lg hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-[#111113] disabled:opacity-70 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  Authenticate
                  <ArrowRight className="absolute right-4 w-4 h-4 transition-transform group-hover:translate-x-1" />
                </>
              )}
            </button>
          </form>
          
          <div className="mt-6 flex items-center gap-2 text-xs text-gray-500">
            <Lock className="w-3 h-3" />
            <span>End-to-End Encrypted HIPAA Gateway</span>
          </div>
        </div>
      </div>
    </div>
  );
}
