"use client";

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [serverUrl, setServerUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:5050"
  );
  const [error, setError] = useState("");
  const router = useRouter();

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (username.trim().toLowerCase() !== "treehacks") {
      setError("Invalid username.");
      return;
    }
    // Store auth + server URL in sessionStorage
    sessionStorage.setItem("authed", "true");
    sessionStorage.setItem("serverUrl", serverUrl.replace(/\/+$/, ""));
    router.push("/chat");
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm flex flex-col gap-4 px-6"
      >
        <h1 className="text-2xl font-bold text-center mb-2">Magic Table</h1>

        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => {
            setUsername(e.target.value);
            setError("");
          }}
          className="border border-gray-300 rounded px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-black min-h-[44px]"
          autoFocus
        />

        <input
          type="text"
          placeholder="Server URL"
          value={serverUrl}
          onChange={(e) => setServerUrl(e.target.value)}
          className="border border-gray-300 rounded px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-black text-gray-500 min-h-[44px]"
        />

        {error && <p className="text-red-600 text-sm text-center">{error}</p>}

        <button
          type="submit"
          className="bg-black text-white rounded px-4 py-2 text-base font-medium hover:bg-gray-800 transition-colors min-h-[44px]"
        >
          Log in
        </button>
      </form>
    </div>
  );
}
