-- Enable UUID extension (only once per project)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop and recreate 'users' table with auto-generated UUIDs
DROP TABLE IF EXISTS users CASCADE;
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL UNIQUE,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Drop and recreate 'chat_sessions' table with UUID default
DROP TABLE IF EXISTS chat_sessions CASCADE;
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Drop and recreate 'chat_messages' table with UUID default
DROP TABLE IF EXISTS chat_messages CASCADE;
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    content TEXT,
    role TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
);
