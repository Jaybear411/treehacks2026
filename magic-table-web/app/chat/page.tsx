"use client";

import { useState, useEffect, useRef, FormEvent } from "react";
import { useRouter } from "next/navigation";

interface Message {
  role: "user" | "assistant";
  text: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [serverUrl, setServerUrl] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Auth gate
  useEffect(() => {
    const authed = sessionStorage.getItem("authed");
    if (authed !== "true") {
      router.replace("/");
      return;
    }
    setServerUrl(
      sessionStorage.getItem("serverUrl") || "http://localhost:5050"
    );
  }, [router]);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    // Add user message
    const userMsg: Message = { role: "user", text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${serverUrl}/api/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();

      const reply =
        data.reply ||
        data.error ||
        "No response from server.";

      setMessages((prev) => [...prev, { role: "assistant", text: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Could not reach server at ${serverUrl}. Is it running?`,
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  async function handleAction(endpoint: string, label: string) {
    if (loading) return;

    setMessages((prev) => [...prev, { role: "user", text: label }]);
    setLoading(true);

    try {
      const res = await fetch(`${serverUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      const data = await res.json();

      const reply =
        data.reply ||
        data.error ||
        "No response from server.";

      setMessages((prev) => [...prev, { role: "assistant", text: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Could not reach server at ${serverUrl}. Is it running?`,
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleLogout() {
    sessionStorage.clear();
    router.replace("/");
  }

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <h1 className="text-lg font-semibold">Magic Table</h1>
        <button
          onClick={handleLogout}
          className="text-sm text-gray-400 hover:text-black transition-colors min-h-[44px] min-w-[44px] px-2 -mr-2"
        >
          Log out
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <p className="text-gray-400 text-sm text-center mt-12">
            Type a command like &quot;fetch the red bottle&quot; or just chat
            with Jarvis.
          </p>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-black text-white"
                  : "bg-gray-100 text-black"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-3 py-2 text-sm text-gray-400">
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Quick-action buttons */}
      <div className="px-4 pt-2 flex gap-2">
        <button
          onClick={() =>
            handleAction("/api/describe", "What's on the table?")
          }
          disabled={loading}
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-base font-medium text-gray-700 hover:bg-gray-50 active:bg-gray-100 transition-colors disabled:opacity-30 min-h-[44px]"
        >
          What&apos;s on the table?
        </button>
        <button
          onClick={() => handleAction("/api/cleanup", "Clean up the table")}
          disabled={loading}
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-base font-medium text-gray-700 hover:bg-gray-50 active:bg-gray-100 transition-colors disabled:opacity-30 min-h-[44px]"
        >
          Cleanup
        </button>
      </div>

      {/* Input */}
      <form
        onSubmit={handleSend}
        className="border-t border-gray-200 px-4 py-3 flex gap-2"
      >
        <input
          ref={inputRef}
          type="text"
          placeholder="Type a command..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          className="flex-1 border border-gray-300 rounded px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-black disabled:opacity-50 min-h-[44px]"
          autoFocus
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-black text-white rounded px-4 py-2 text-base font-medium hover:bg-gray-800 transition-colors disabled:opacity-30 min-h-[44px]"
        >
          Send
        </button>
      </form>
    </div>
  );
}
