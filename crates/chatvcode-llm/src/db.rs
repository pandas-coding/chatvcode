//! Database persistence for conversation history.
//!
//! Provides SQLite-based storage for chat sessions, replacing JSON file storage
//! for better query capabilities and performance.
//!
//! # Example
//!
//! ```ignore
//! use chatvcode_llm::db::SessionDatabase;
//! use chatvcode_llm::{ChatSession, ChatTemplate};
//!
//! let db = SessionDatabase::open("~/.chatvcode/sessions.db")?;
//!
//! // Save a session
//! let session = ChatSession::new(ChatTemplate::ChatML);
//! db.save_session(&session)?;
//!
//! // List all sessions
//! let sessions = db.list_sessions()?;
//!
//! // Load a specific session
//! let loaded = db.load_session("session-id")?;
//! ```

use std::path::Path;
use std::time::SystemTime;

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::chat::message::ChatMessage;
use crate::chat::session::ChatSession;
use crate::chat::template::ChatTemplate;
use crate::error::{LlmError, LlmResult};

/// A persisted session record from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Unique session identifier.
    pub id: String,
    /// Session creation timestamp (Unix epoch seconds).
    pub created_at: i64,
    /// Last activity timestamp (Unix epoch seconds).
    pub updated_at: i64,
    /// Chat template used.
    pub template: String,
    /// System prompt (if any).
    pub system_prompt: Option<String>,
    /// Number of messages in the session.
    pub message_count: i32,
    /// Number of completed turns.
    pub turn_count: i32,
}

/// SQLite database for persisting chat sessions.
pub struct SessionDatabase {
    conn: Connection,
}

impl SessionDatabase {
    /// Open or create a session database at the given path.
    ///
    /// Creates the database file and schema if they don't exist.
    pub fn open(path: impl AsRef<Path>) -> LlmResult<Self> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                LlmError::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to create database directory: {e}"),
                ))
            })?;
        }

        let conn = Connection::open(path).map_err(|e| {
            LlmError::Internal(format!("Failed to open database at {}: {e}", path.display()))
        })?;

        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Open an in-memory database (useful for testing).
    pub fn open_in_memory() -> LlmResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| LlmError::Internal(format!("Failed to open in-memory database: {e}")))?;

        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> LlmResult<()> {
        self.conn
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    template TEXT NOT NULL,
                    system_prompt TEXT,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    turn_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, position);
                ",
            )
            .map_err(|e| LlmError::Internal(format!("Failed to initialize database schema: {e}")))?;

        Ok(())
    }

    /// Save a chat session to the database.
    ///
    /// If a session with the same ID exists, it is updated.
    pub fn save_session(&self, session: &ChatSession) -> LlmResult<()> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let template_name = session
            .template()
            .template_name()
            .unwrap_or("auto")
            .to_string();

        self.conn
            .execute(
                "INSERT OR REPLACE INTO sessions (id, created_at, updated_at, template, system_prompt, message_count, turn_count)
                 VALUES (?1, COALESCE((SELECT created_at FROM sessions WHERE id = ?1), ?2), ?2, ?3, ?4, ?5, ?6)",
                params![
                    session.session_id(),
                    now,
                    template_name,
                    session.get_system_prompt(),
                    session.len() as i32,
                    session.turn_count() as i32,
                ],
            )
            .map_err(|e| LlmError::Internal(format!("Failed to save session: {e}")))?;

        // Delete existing messages and re-insert
        self.conn
            .execute("DELETE FROM messages WHERE session_id = ?1", params![session.session_id()])
            .map_err(|e| LlmError::Internal(format!("Failed to clear messages: {e}")))?;

        for (pos, msg) in session.messages().iter().enumerate() {
            self.conn
                .execute(
                    "INSERT INTO messages (session_id, role, content, position, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![session.session_id(), msg.role, msg.content, pos as i32, now],
                )
                .map_err(|e| LlmError::Internal(format!("Failed to save message: {e}")))?;
        }

        Ok(())
    }

    /// Load a session from the database by ID.
    pub fn load_session(&self, session_id: &str, template: ChatTemplate) -> LlmResult<Option<ChatSession>> {
        let mut stmt = self
            .conn
            .prepare("SELECT system_prompt FROM sessions WHERE id = ?1")
            .map_err(|e| LlmError::Internal(format!("Failed to prepare query: {e}")))?;

        let system_prompt: Option<String> = match stmt.query_row(params![session_id], |row| row.get(0)) {
            Ok(p) => p,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(LlmError::Internal(format!("Failed to load session: {e}"))),
        };

        let mut session = ChatSession::new(template);
        if let Some(prompt) = system_prompt {
            session.set_system_prompt(Some(prompt));
        }

        // Load messages
        let mut msg_stmt = self
            .conn
            .prepare("SELECT role, content FROM messages WHERE session_id = ?1 ORDER BY position")
            .map_err(|e| LlmError::Internal(format!("Failed to prepare message query: {e}")))?;

        let messages: Vec<ChatMessage> = msg_stmt
            .query_map(params![session_id], |row| {
                Ok(ChatMessage {
                    role: row.get(0)?,
                    content: row.get(1)?,
                })
            })
            .map_err(|e| LlmError::Internal(format!("Failed to query messages: {e}")))?
            .filter_map(|r| r.ok())
            .collect();

        for msg in messages {
            session.add_message(msg);
        }

        Ok(Some(session))
    }

    /// List all saved sessions, ordered by most recently updated.
    pub fn list_sessions(&self) -> LlmResult<Vec<SessionRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, created_at, updated_at, template, system_prompt, message_count, turn_count
                 FROM sessions ORDER BY updated_at DESC",
            )
            .map_err(|e| LlmError::Internal(format!("Failed to prepare query: {e}")))?;

        let records = stmt
            .query_map([], |row| {
                Ok(SessionRecord {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    updated_at: row.get(2)?,
                    template: row.get(3)?,
                    system_prompt: row.get(4)?,
                    message_count: row.get(5)?,
                    turn_count: row.get(6)?,
                })
            })
            .map_err(|e| LlmError::Internal(format!("Failed to list sessions: {e}")))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(records)
    }

    /// Delete a session by ID.
    pub fn delete_session(&self, session_id: &str) -> LlmResult<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM sessions WHERE id = ?1", params![session_id])
            .map_err(|e| LlmError::Internal(format!("Failed to delete session: {e}")))?;

        Ok(rows > 0)
    }

    /// Delete all sessions.
    pub fn clear_all(&self) -> LlmResult<usize> {
        let rows = self
            .conn
            .execute("DELETE FROM sessions", [])
            .map_err(|e| LlmError::Internal(format!("Failed to clear sessions: {e}")))?;

        Ok(rows)
    }

    /// Get the total number of saved sessions.
    pub fn session_count(&self) -> LlmResult<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .map_err(|e| LlmError::Internal(format!("Failed to count sessions: {e}")))?;

        Ok(count as usize)
    }

    /// Search sessions by content (message text).
    pub fn search_sessions(&self, query: &str) -> LlmResult<Vec<SessionRecord>> {
        let pattern = format!("%{query}%");
        let mut stmt = self
            .conn
            .prepare(
                "SELECT DISTINCT s.id, s.created_at, s.updated_at, s.template,
                        s.system_prompt, s.message_count, s.turn_count
                 FROM sessions s
                 JOIN messages m ON s.id = m.session_id
                 WHERE m.content LIKE ?1
                 ORDER BY s.updated_at DESC",
            )
            .map_err(|e| LlmError::Internal(format!("Failed to prepare search: {e}")))?;

        let records = stmt
            .query_map(params![pattern], |row| {
                Ok(SessionRecord {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    updated_at: row.get(2)?,
                    template: row.get(3)?,
                    system_prompt: row.get(4)?,
                    message_count: row.get(5)?,
                    turn_count: row.get(6)?,
                })
            })
            .map_err(|e| LlmError::Internal(format!("Failed to search sessions: {e}")))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_open_in_memory() {
        let db = SessionDatabase::open_in_memory().unwrap();
        assert_eq!(db.session_count().unwrap(), 0);
    }

    #[test]
    fn test_save_and_load_session() {
        let db = SessionDatabase::open_in_memory().unwrap();

        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("You are helpful.");
        session.add_user_message("Hello");
        session.add_assistant_message("Hi there!");

        db.save_session(&session).unwrap();
        assert_eq!(db.session_count().unwrap(), 1);

        let loaded = db.load_session(session.session_id(), ChatTemplate::ChatML).unwrap();
        assert!(loaded.is_some());

        let loaded = loaded.unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get_system_prompt(), Some("You are helpful."));
    }

    #[test]
    fn test_list_sessions() {
        let db = SessionDatabase::open_in_memory().unwrap();

        let mut s1 = ChatSession::new(ChatTemplate::ChatML);
        s1.add_user_message("First session");
        db.save_session(&s1).unwrap();

        let mut s2 = ChatSession::new(ChatTemplate::Llama3);
        s2.add_user_message("Second session");
        db.save_session(&s2).unwrap();

        let sessions = db.list_sessions().unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[test]
    fn test_delete_session() {
        let db = SessionDatabase::open_in_memory().unwrap();

        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.add_user_message("Test");
        db.save_session(&session).unwrap();

        assert_eq!(db.session_count().unwrap(), 1);
        assert!(db.delete_session(session.session_id()).unwrap());
        assert_eq!(db.session_count().unwrap(), 0);
    }

    #[test]
    fn test_clear_all() {
        let db = SessionDatabase::open_in_memory().unwrap();

        for i in 0..5 {
            let mut s = ChatSession::new(ChatTemplate::ChatML);
            s.add_user_message(format!("Session {i}"));
            db.save_session(&s).unwrap();
        }

        assert_eq!(db.session_count().unwrap(), 5);
        let cleared = db.clear_all().unwrap();
        assert_eq!(cleared, 5);
        assert_eq!(db.session_count().unwrap(), 0);
    }

    #[test]
    fn test_search_sessions() {
        let db = SessionDatabase::open_in_memory().unwrap();

        let mut s1 = ChatSession::new(ChatTemplate::ChatML);
        s1.add_user_message("Hello Rust world");
        db.save_session(&s1).unwrap();

        let mut s2 = ChatSession::new(ChatTemplate::ChatML);
        s2.add_user_message("Python is great");
        db.save_session(&s2).unwrap();

        let results = db.search_sessions("Rust").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, s1.session_id());
    }

    #[test]
    fn test_load_nonexistent_session() {
        let db = SessionDatabase::open_in_memory().unwrap();
        let result = db.load_session("nonexistent-id", ChatTemplate::ChatML).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_update_existing_session() {
        let db = SessionDatabase::open_in_memory().unwrap();

        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.add_user_message("First message");
        db.save_session(&session).unwrap();

        session.add_assistant_message("Response");
        session.add_user_message("Second message");
        db.save_session(&session).unwrap();

        assert_eq!(db.session_count().unwrap(), 1);

        let loaded = db.load_session(session.session_id(), ChatTemplate::ChatML).unwrap().unwrap();
        assert_eq!(loaded.len(), 3);
    }
}
