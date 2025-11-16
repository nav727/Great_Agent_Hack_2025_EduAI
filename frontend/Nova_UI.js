import React, { useState, useRef, useEffect } from 'react';
import { Send, X, Plus, MessageSquare, Menu, Search, Bot, User, Sparkles } from 'lucide-react';

export default function NovaChatbot() {
  const [mainInput, setMainInput] = useState('');
  const [keywordInput, setKeywordInput] = useState('');
  const [keywords, setKeywords] = useState([]);
  const [showKeywordBar, setShowKeywordBar] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const messagesEndRef = useRef(null);
  const mainInputRef = useRef(null);

  const handleMainInputChange = (e) => {
    const value = e.target.value;
    setMainInput(value);
    setShowKeywordBar(value.length > 0);
  };

  const handleKeywordSubmit = (e) => {
    if (e.key === 'Enter' && keywordInput.trim()) {
      setKeywords([...keywords, keywordInput.trim()]);
      setKeywordInput('');
    }
  };

  const removeKeyword = (index) => {
    setKeywords(keywords.filter((_, i) => i !== index));
  };

  const handleMainSubmit = () => {
    if (mainInput.trim()) {
      const newMessage = {
        id: Date.now(),
        query: mainInput,
        keywords: [...keywords],
        response: "I'm NOVA, your AI assistant. This is a sample response demonstrating the improved chat interface. In a production environment, this would connect to your backend API to provide real, intelligent responses based on your query and keywords.",
        timestamp: new Date(),
        isUser: true
      };
      
      const updatedMessages = [...messages, newMessage];
      setMessages(updatedMessages);
      
      // Simulate AI typing
      setIsTyping(true);
      setTimeout(() => setIsTyping(false), 1500);
      
      if (!currentChatId) {
        const newChatId = Date.now();
        const newChat = {
          id: newChatId,
          name: mainInput.slice(0, 40) + (mainInput.length > 40 ? '...' : ''),
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
      
      setMainInput('');
      setKeywordInput('');
      setKeywords([]);
      setShowKeywordBar(false);
      mainInputRef.current?.focus();
    }
  };

  const handleMainInputKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleMainSubmit();
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setCurrentChatId(null);
    setMainInput('');
    setKeywordInput('');
    setKeywords([]);
    mainInputRef.current?.focus();
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

  const filteredChats = chats.filter(chat =>
    chat.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
  };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex overflow-hidden">
      {/* Sidebar */}
      <div className={`bg-slate-900/50 backdrop-blur-xl border-r border-slate-800/50 flex flex-col transition-all duration-300 ease-in-out ${
        sidebarOpen ? 'w-72' : 'w-0'
      } overflow-hidden`}>
        <div className={`flex flex-col h-full transition-opacity duration-200 ${sidebarOpen ? 'opacity-100' : 'opacity-0'}`}>
          {/* Sidebar Header */}
          <div className="p-4 border-b border-slate-800/50">
            <button
              onClick={startNewChat}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 font-medium"
            >
              <Plus size={20} />
              <span>New Chat</span>
            </button>
          </div>

          {/* Search */}
          <div className="p-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search conversations..."
                className="w-full bg-slate-800/50 text-slate-200 pl-10 pr-4 py-2 rounded-lg border border-slate-700/50 focus:outline-none focus:border-blue-500/50 transition-all text-sm placeholder-slate-500"
              />
            </div>
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto px-4 pb-4">
            <h3 className="text-slate-400 text-xs font-semibold mb-3 uppercase tracking-wider px-2">Recent Chats</h3>
            <div className="space-y-1">
              {filteredChats.length === 0 ? (
                <p className="text-slate-500 text-sm text-center py-8">No chats found</p>
              ) : (
                filteredChats.map((chat) => (
                  <div
                    key={chat.id}
                    className={`p-3 rounded-xl cursor-pointer transition-all group relative ${
                      currentChatId === chat.id
                        ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30'
                        : 'text-slate-300 hover:bg-slate-800/50 border border-transparent'
                    }`}
                    onClick={() => loadChat(chat.id)}
                  >
                    <div className="flex items-start gap-3">
                      <MessageSquare size={16} className="mt-1 flex-shrink-0 opacity-70" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{chat.name}</p>
                        <p className="text-xs opacity-60 mt-0.5">
                          {new Date(chat.createdAt).toLocaleDateString('en-US', { 
                            month: 'short', 
                            day: 'numeric' 
                          })}
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteChat(chat.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 hover:text-red-400"
                      >
                        <X size={16} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex justify-between items-center px-6 py-4 border-b border-slate-800/50 bg-slate-900/30 backdrop-blur-xl">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-slate-400 hover:text-blue-400 transition-colors p-2 rounded-lg hover:bg-slate-800/50"
          >
            <Menu size={22} />
          </button>
          <div className="flex items-center gap-3 bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-2.5 rounded-xl shadow-lg">
            <Sparkles size={20} className="text-white" />
            <span className="text-white font-bold text-2xl tracking-wide">NOVA</span>
          </div>
          <div className="w-10"></div>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-4 py-6">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center py-20">
                <div className="bg-gradient-to-br from-blue-600/20 to-purple-600/20 p-6 rounded-full mb-6">
                  <Bot size={48} className="text-blue-400" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">Welcome to NOVA</h2>
                <p className="text-slate-400 max-w-md">
                  Start a conversation by typing your question below. Add keywords to refine your search.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((message) => (
                  <div key={message.id} className="space-y-4">
                    {/* User Query */}
                    <div className="flex justify-end gap-3">
                      <div className="flex flex-col items-end max-w-2xl">
                        <div className="bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-2xl rounded-tr-sm px-5 py-3 shadow-lg">
                          <p className="text-sm leading-relaxed">{message.query}</p>
                          {message.keywords.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-blue-500/30">
                              {message.keywords.map((keyword, idx) => (
                                <span
                                  key={idx}
                                  className="bg-blue-800/50 px-2.5 py-1 rounded-lg text-xs font-medium backdrop-blur-sm"
                                >
                                  {keyword}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                        <span className="text-xs text-slate-500 mt-1.5 px-2">
                          {formatTime(message.timestamp)}
                        </span>
                      </div>
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center flex-shrink-0 shadow-lg">
                        <User size={16} className="text-white" />
                      </div>
                    </div>

                    {/* AI Response */}
                    <div className="flex justify-start gap-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center flex-shrink-0 shadow-lg">
                        <Bot size={16} className="text-white" />
                      </div>
                      <div className="flex flex-col items-start max-w-2xl">
                        <div className="bg-slate-800/50 backdrop-blur-sm text-slate-100 rounded-2xl rounded-tl-sm px-5 py-3 shadow-lg border border-slate-700/50">
                          <p className="text-sm leading-relaxed">{message.response}</p>
                        </div>
                        <span className="text-xs text-slate-500 mt-1.5 px-2">
                          {formatTime(message.timestamp)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                  <div className="flex justify-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center flex-shrink-0 shadow-lg">
                      <Bot size={16} className="text-white" />
                    </div>
                    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl rounded-tl-sm px-5 py-4 shadow-lg border border-slate-700/50">
                      <div className="flex gap-1.5">
                        <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Input Section */}
        <div className="border-t border-slate-800/50 bg-slate-900/30 backdrop-blur-xl px-4 py-4">
          <div className="max-w-4xl mx-auto">
            {/* Keywords Display */}
            {keywords.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {keywords.map((keyword, index) => (
                  <div
                    key={index}
                    className="bg-slate-800/70 backdrop-blur-sm border border-slate-700/50 text-slate-200 px-3 py-1.5 rounded-full flex items-center gap-2 animate-slideIn shadow-sm"
                  >
                    <span className="text-sm font-medium">{keyword}</span>
                    <button
                      onClick={() => removeKeyword(index)}
                      className="hover:text-red-400 transition-colors hover:scale-110"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Input Container */}
            <div className="flex gap-2 items-end">
              {/* Main Input */}
              <div className="flex-1 relative">
                <textarea
                  ref={mainInputRef}
                  value={mainInput}
                  onChange={handleMainInputChange}
                  onKeyDown={handleMainInputKeyDown}
                  placeholder="Ask NOVA anything..."
                  rows={1}
                  className="w-full bg-slate-800/50 backdrop-blur-sm text-slate-100 px-5 py-3.5 pr-14 rounded-2xl border border-slate-700/50 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 transition-all resize-none placeholder-slate-500"
                  style={{ minHeight: '52px', maxHeight: '120px' }}
                />
                <button
                  onClick={handleMainSubmit}
                  disabled={!mainInput.trim()}
                  className={`absolute right-3 bottom-3 p-2.5 rounded-xl transition-all ${
                    mainInput.trim()
                      ? 'bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white shadow-lg shadow-blue-500/30'
                      : 'bg-slate-700/50 text-slate-500 cursor-not-allowed'
                  }`}
                >
                  <Send size={18} />
                </button>
              </div>

              {/* Keyword Bar */}
              <div
                className={`overflow-hidden transition-all duration-300 ease-out ${
                  showKeywordBar ? 'w-44 opacity-100' : 'w-0 opacity-0'
                }`}
              >
                <input
                  type="text"
                  value={keywordInput}
                  onChange={(e) => setKeywordInput(e.target.value)}
                  onKeyDown={handleKeywordSubmit}
                  placeholder="Add keyword..."
                  className="w-full bg-slate-800/50 backdrop-blur-sm text-slate-100 px-4 py-3.5 rounded-2xl border border-slate-700/50 focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition-all text-sm placeholder-slate-500"
                />
              </div>
            </div>

            {/* Hint Text */}
            <p className="text-xs text-slate-500 mt-2 text-center">
              Press <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700">Enter</kbd> to send • 
              <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700 ml-1">Shift + Enter</kbd> for new line
            </p>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(-10px) scale(0.95);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
        .animate-slideIn {
          animation: slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
          width: 8px;
        }
        ::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.3);
        }
        ::-webkit-scrollbar-thumb {
          background: rgba(71, 85, 105, 0.5);
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(71, 85, 105, 0.7);
        }
      `}</style>
    </div>
  );
}
