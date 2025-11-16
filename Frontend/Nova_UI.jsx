import React, { useState, useRef, useEffect } from 'react';
import { Send, X, Plus, MessageSquare, Menu } from 'lucide-react';

export default function NovaChatbot() {
  const [mainInput, setMainInput] = useState('');
  const [keywordInput, setKeywordInput] = useState('');
  const [keywords, setKeywords] = useState([]);
  const [showKeywordBar, setShowKeywordBar] = useState(false);
  const [keywordError, setKeywordError] = useState('');
  const [keywordShake, setKeywordShake] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [contentVisible, setContentVisible] = useState(true);
  const messagesEndRef = useRef(null);

  const formatAIResponse = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    const blocks = [];
    let i = 0;

    while (i < lines.length) {
      const raw = lines[i].trim();
      if (raw.length === 0) {
        i += 1;
        continue;
      }

      // Numbered section as a small header, e.g., "1. Topic"
      const numHeader = raw.match(/^(\d+)\.\s+(.*)$/);
      if (numHeader) {
        blocks.push(
          <h4 key={`h-${i}`} className="text-slate-200 font-semibold mt-3 text-base md:text-lg">
            {numHeader[1]}. {numHeader[2]}
          </h4>
        );
        i += 1;
        continue;
      }

      // Gather bullet list items starting with * or -
      if (/^(\*|-)\s+/.test(raw)) {
        const items = [];
        while (i < lines.length) {
          const li = lines[i].trim();
          if (!/^(\*|-)\s+/.test(li)) break;
          items.push(li.replace(/^(\*|-)\s+/, ''));
          i += 1;
        }
        blocks.push(
          <ul key={`ul-${i}`} className="list-disc pl-5 space-y-1">
            {items.map((it, idx) => (
              <li key={idx} className="text-slate-200 leading-relaxed">{it}</li>
            ))}
          </ul>
        );
        continue;
      }

      // "Section: Title" => render title bold and put following as paragraph
      const labelHeader = raw.match(/^([A-Za-z][\w\s]+):\s*(.*)$/);
      if (labelHeader) {
        blocks.push(
          <p key={`lh-${i}`} className="text-slate-200 leading-relaxed">
            <span className="font-semibold">{labelHeader[1]}: </span>
            {labelHeader[2]}
          </p>
        );
        i += 1;
        continue;
      }

      // Default paragraph; also merge subsequent non-empty, non-bullet lines
      const paraParts = [raw];
      let j = i + 1;
      while (j < lines.length) {
        const nxt = lines[j].trim();
        if (
          nxt.length === 0 ||
          /^(\*|-)\s+/.test(nxt) ||
          /^(\d+)\.\s+/.test(nxt)
        ) {
          break;
        }
        paraParts.push(nxt);
        j += 1;
      }
      blocks.push(
        <p key={`p-${i}`} className="text-slate-200 leading-relaxed">
          {paraParts.join(' ')}
        </p>
      );
      i = j;
    }

    return <div className="space-y-2">{blocks}</div>;
  };

  // Typing effect support
  const typingTimersRef = useRef({});

  const startTypingAnimation = (messageId, fullText) => {
    const totalLength = fullText.length;
    if (totalLength === 0) {
      setMessages(prev =>
        prev.map(m => (m.id === messageId ? { ...m, response: '', isTyping: false, typingTarget: '' } : m))
      );
      return;
    }
    const step = Math.max(1, Math.ceil(totalLength / 120)); // around ~120 steps
    let index = 0;
    // Clear any existing timer for this message
    if (typingTimersRef.current[messageId]) {
      clearInterval(typingTimersRef.current[messageId]);
    }
    const interval = setInterval(() => {
      index = Math.min(totalLength, index + step);
      const chunk = fullText.slice(0, index);
      setMessages(prev =>
        prev.map(m =>
          m.id === messageId ? { ...m, response: chunk, isTyping: index < totalLength, typingTarget: fullText } : m
        )
      );
      if (index >= totalLength) {
        clearInterval(interval);
        delete typingTimersRef.current[messageId];
      }
    }, 16); // ~60fps
    typingTimersRef.current[messageId] = interval;
  };

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      Object.values(typingTimersRef.current).forEach((t) => clearInterval(t));
      typingTimersRef.current = {};
    };
  }, []);

  const handleMainInputChange = (e) => {
    const value = e.target.value;
    setMainInput(value);
    setShowKeywordBar(value.length > 0);
  };

  const handleKeywordSubmit = (e) => {
    if (e.key === 'Enter' && keywordInput.trim()) {
      const candidate = keywordInput.trim();
      const lowerCandidate = candidate.toLowerCase();
      const existing = new Set(keywords.map(k => String(k).toLowerCase()));
      if (existing.has(lowerCandidate)) {
        setKeywordError('Keyword already added');
        setKeywordShake(true);
        setTimeout(() => setKeywordShake(false), 450);
        setTimeout(() => setKeywordError(''), 1500);
        return;
      }
      setKeywords([...keywords, candidate]);
      setKeywordInput('');
    }
  };

  const removeKeyword = (index) => {
    setKeywords(keywords.filter((_, i) => i !== index));
  };

  const handleMainSubmit = async () => {
    if (mainInput.trim()) {
      let createdChatId = null;
      const newMessage = {
        id: Date.now(),
        query: mainInput,
        keywords: [...keywords],
        response: "",
        isLoading: true,
        isTyping: false,
        typingTarget: "",
        timestamp: new Date()
      };
      
      const updatedMessages = [...messages, newMessage];
      setMessages(updatedMessages);
      
      if (!currentChatId) {
        const newChatId = Date.now();
        createdChatId = newChatId;
        const newChat = {
          id: newChatId,
          name: mainInput.slice(0, 30) + (mainInput.length > 30 ? '...' : ''),
          messages: updatedMessages,
          createdAt: new Date()
        };
        setChats([newChat, ...chats]);
        setCurrentChatId(newChatId);
      } else {
        const updatedChats = chats.map(chat =>
          chat.id === currentChatId ? { ...chat, messages: updatedMessages } : chat
        );
        setChats(updatedChats);
      }
      
      // Reset inputs
      setMainInput('');
      setKeywordInput('');
      setKeywords([]);
      setShowKeywordBar(false);

      // Call backend
      try {
        const res = await fetch('http://127.0.0.1:8000/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: newMessage.query, keywords: newMessage.keywords }),
        });
        const data = await res.json();
        const aiResponse = data?.answer ?? 'No response received.';

        // Stop loading; start typing animation
        setMessages(prev =>
          prev.map(m =>
            m.id === newMessage.id
              ? { ...m, isLoading: false, isTyping: true, typingTarget: aiResponse, response: '' }
              : m
          )
        );
        startTypingAnimation(newMessage.id, aiResponse);

        // Update chats with new response
        const activeChatId = currentChatId || createdChatId;
        if (activeChatId) {
          setChats(prev =>
            prev.map(chat =>
              chat.id === activeChatId
                ? {
                    ...chat,
                    messages: chat.messages.map(m =>
                      m.id === newMessage.id
                        ? { ...m, isLoading: false, isTyping: true, typingTarget: aiResponse, response: '' }
                        : m
                    ),
                  }
                : chat
            )
          );
        }
      } catch (e) {
        const errText = 'Error contacting backend.';
        setMessages(prev =>
          prev.map(m => (m.id === newMessage.id ? { ...m, response: errText, isLoading: false, isTyping: false } : m))
        );
      }
    }
  };

  const handleMainInputKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleMainSubmit();
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setCurrentChatId(null);
    setMainInput('');
    setKeywordInput('');
    setKeywords([]);
  };

  const loadChat = (chatId) => {
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setMessages(chat.messages);
      setCurrentChatId(chatId);
    }
  };

  const deleteChat = (chatId) => {
    setChats(chats.filter(c => c.id !== chatId));
    if (currentChatId === chatId) {
      startNewChat();
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (sidebarOpen) {
      const timer = setTimeout(() => setContentVisible(true), 200);
      return () => clearTimeout(timer);
    } else {
      setContentVisible(false);
    }
  }, [sidebarOpen]);

  return (
    <div className="h-screen bg-black flex overflow-hidden">
      {/* Sidebar */}
      <div className={`bg-slate-950 flex flex-col transition-all duration-500 ease-in-out overflow-hidden ${
        sidebarOpen ? 'w-64 p-4' : 'w-0 p-0'
      }`}>
        {/* New Chat Button - Hidden during animation */}
        {contentVisible && (
          <button
            onClick={startNewChat}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg flex items-center justify-center gap-2 mb-6 transition-colors flex-shrink-0"
          >
            <Plus size={20} />
            <span>New Chat</span>
          </button>
        )}

        {/* Chat History - Hidden during animation */}
        {contentVisible && (
          <div className="flex-1 overflow-y-auto min-w-0">
            <h3 className="text-slate-400 text-xs font-bold mb-3 uppercase tracking-wider">Chat History</h3>
            <div className="space-y-2">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`p-3 rounded-lg cursor-pointer transition-all group ${
                    currentChatId === chat.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                  onClick={() => loadChat(chat.id)}
                >
                  <div className="flex items-start gap-2">
                    <MessageSquare size={16} className="mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">{chat.name}</p>
                      <p className="text-xs opacity-75">
                        {new Date(chat.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteChat(chat.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header with NOVA Logo and Toggle */}
        <div className="flex justify-between items-center pt-6 pb-6 border-b border-slate-800 app-gutter flex-shrink-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-white hover:text-blue-400 transition-colors"
          >
            <Menu size={24} />
          </button>
          <div className="bg-blue-600 px-6 py-3 rounded-lg">
            <span className="text-white font-bold text-3xl tracking-wider">NOVA</span>
          </div>
          <div className="w-6"></div>
        </div>

        {/* Messages Container - Scrollable */}
        <div className="flex-1 overflow-y-auto app-gutter py-6">
          <div className="max-w-5xl mx-auto">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-500">
                <p>Start a new conversation...</p>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((message) => (
                  <div key={message.id} className="space-y-3">
                    {/* User Query - Message Style */}
                    <div className="flex justify-end">
                      <div className="bg-blue-700 text-white rounded-lg p-3 max-w-xs lg:max-w-md break-words">
                        <p className="text-sm leading-relaxed">{message.query}</p>
                        {message.keywords.length > 0 && (
                          <div className="flex flex-wrap gap-2 mt-2 pt-2 border-t border-blue-600">
                            {message.keywords.map((keyword, idx) => (
                              <span
                                key={idx}
                                className="bg-blue-800 px-2 py-1 rounded text-xs"
                              >
                                "{keyword}"
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* AI Response - Message Style */}
                    <div className="flex justify-start">
                      <div className="bg-slate-800 text-white rounded-lg p-3 max-w-xs lg:max-w-2xl break-words">
                        {message.isLoading ? (
                          <div className="flex items-center gap-1 py-1">
                            <span className="loader-dot animate-bounce-1"></span>
                            <span className="loader-dot animate-bounce-2"></span>
                            <span className="loader-dot animate-bounce-3"></span>
                          </div>
                        ) : (
                          <div className="text-sm">
                            {formatAIResponse(message.response)}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Input Section - Fixed at Bottom */}
        <div className="border-t border-slate-800 bg-black app-gutter py-6 flex-shrink-0">
          <div className="max-w-5xl mx-auto">
            {/* Keywords Display */}
            {keywords.length > 0 && (
              <div className="mb-4 flex flex-wrap gap-2">
                {keywords.map((keyword, index) => (
                  <div
                    key={index}
                    className="bg-slate-800 text-white px-3 py-1 rounded-full flex items-center gap-2 animate-fadeIn"
                  >
                    <span className="text-sm">{keyword}</span>
                    <button
                      onClick={() => removeKeyword(index)}
                      className="hover:text-red-400 transition-colors"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Search Container */}
            <div className="flex gap-2 items-center">
              {/* Main Input */}
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={mainInput}
                  onChange={handleMainInputChange}
                  onKeyDown={handleMainInputKeyDown}
                  placeholder="Enter your prompt..."
                  className="w-full bg-slate-900 text-white px-4 py-3 rounded-lg border border-slate-700 focus:outline-none focus:border-blue-500 transition-all"
                />
                <button
                  onClick={handleMainSubmit}
                  className="absolute right-3 top-1/2 -translate-y-1/2 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-lg transition-colors"
                >
                  <Send size={20} />
                </button>
              </div>

              {/* Keyword Bar - Slides in/out */}
              <div
                className={`overflow-hidden transition-all duration-300 ease-out ${
                  showKeywordBar ? 'w-40 opacity-100' : 'w-0 opacity-0'
                }`}
              >
                <div className={`relative px-1 ${keywordShake ? 'animate-shake' : ''}`}>
                  <input
                    type="text"
                    value={keywordInput}
                    onChange={(e) => setKeywordInput(e.target.value)}
                    onKeyDown={handleKeywordSubmit}
                    placeholder="Add keyword..."
                    className={`w-full bg-slate-900 text-white px-4 py-3 rounded-lg border ${keywordError ? 'border-red-500' : 'border-slate-700'} focus:outline-none focus:border-blue-500 transition-all text-sm`}
                    aria-invalid={!!keywordError}
                  />
                  {keywordError && (
                    <div className="absolute left-0 -bottom-7 bg-red-600 text-white text-xs px-2 py-1 rounded shadow">
                      {keywordError}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: scale(0.8);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}


