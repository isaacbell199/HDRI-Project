//! Database operations for NexaStory using SQLite

use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use anyhow::Result;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use crate::models::*;

/// Thread-safe global database pool storage
static DB_POOL: Lazy<RwLock<Option<SqlitePool>>> = Lazy::new(|| RwLock::new(None));

/// Initialize the database connection and create tables
pub async fn init_database(database_url: &str) -> Result<()> {
    // Configure connection pool with optimized settings
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .min_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(30))
        .idle_timeout(Some(std::time::Duration::from_secs(600)))
        .connect(database_url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect to database at '{}': {}", database_url, e))?;

    // Create tables
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            cover_image TEXT,
            genre TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            order_index INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'draft',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS characters (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            age TEXT,
            gender TEXT,
            role TEXT,
            occupation TEXT,
            appearance TEXT,
            distinguishing_features TEXT,
            personality TEXT,
            traits TEXT,
            flaws TEXT,
            fears TEXT,
            desires TEXT,
            background TEXT,
            relationships TEXT,
            skills TEXT,
            arc TEXT,
            motivation TEXT,
            conflicts TEXT,
            speech_pattern TEXT,
            catchphrases TEXT,
            notes TEXT,
            avatar TEXT,
            color TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS locations (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            location_type TEXT,
            description TEXT,
            atmosphere TEXT,
            features TEXT,
            history TEXT,
            notes TEXT,
            image TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS lore_notes (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            content TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS project_settings (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL UNIQUE,
            target_word_count INTEGER DEFAULT 50000,
            daily_word_goal INTEGER DEFAULT 1000,
            auto_save INTEGER DEFAULT 1,
            auto_save_interval INTEGER DEFAULT 30000,
            context_paragraphs INTEGER DEFAULT 3,
            genres TEXT,
            themes TEXT,
            target_audience TEXT,
            writing_style TEXT,
            narrative_pov TEXT,
            content_rating TEXT,
            content_warnings TEXT,
            tone_preferences TEXT,
            time_period TEXT,
            world_type TEXT,
            language TEXT,
            language_style TEXT,
            adult_content TEXT,
            adult_intensity TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS generation_presets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            preset_type TEXT NOT NULL,
            positive_prompt TEXT,
            negative_prompt TEXT,
            selected_tone TEXT,
            custom_tone_instruction TEXT,
            custom_style_name TEXT,
            custom_style_instruction TEXT,
            custom_genre_name TEXT,
            custom_genre_instruction TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .execute(&pool)
    .await?;

    // Run migrations to add missing columns (robust approach)
    run_migrations(&pool).await?;

    // Store pool globally in thread-safe manner
    {
        let mut db_pool = DB_POOL.write();
        *db_pool = Some(pool);
    }

    Ok(())
}

/// Get the database pool
pub fn get_pool() -> Result<SqlitePool> {
    let db_pool = DB_POOL.read();
    db_pool.clone().ok_or_else(|| anyhow::anyhow!("Database not initialized. Please restart the application."))
}

/// Close all database connections (used before backup restore) (C5)
pub fn close_pool() {
    let mut db_pool = DB_POOL.write();
    if let Some(pool) = db_pool.take() {
        // Drop the pool, which closes all connections
        drop(pool);
    }
}

/// Valid table names for migration checks (M3)
const VALID_TABLES: &[&str] = &[
    "projects", "chapters", "characters", "locations",
    "lore_notes", "project_settings", "generation_presets"
];

/// Check if a column exists in a table
async fn column_exists(pool: &SqlitePool, table: &str, column: &str) -> Result<bool> {
    // Validate table name to prevent SQL injection (M3)
    if !VALID_TABLES.contains(&table) {
        return Err(anyhow::anyhow!("Invalid table name: {}", table));
    }

    let rows = sqlx::query(
        format!("PRAGMA table_info({})", table).as_str()
    )
    .fetch_all(pool)
    .await?;

    for row in rows {
        let col_name: String = row.get("name");
        if col_name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Run database migrations to add missing columns
async fn run_migrations(pool: &SqlitePool) -> Result<()> {
    log::info!("Running database migrations...");

    // Migration: Add order_index column to chapters if it doesn't exist
    if !column_exists(pool, "chapters", "order_index").await? {
        log::info!("Adding order_index column to chapters...");
        sqlx::query("ALTER TABLE chapters ADD COLUMN order_index INTEGER DEFAULT 0")
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to add order_index column: {}", e))?;
        log::info!("Successfully added order_index column");
    } else {
        log::info!("order_index column already exists");
    }

    // Migration: Add word_count column to chapters if it doesn't exist
    if !column_exists(pool, "chapters", "word_count").await? {
        log::info!("Adding word_count column to chapters...");
        sqlx::query("ALTER TABLE chapters ADD COLUMN word_count INTEGER DEFAULT 0")
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to add word_count column: {}", e))?;
        log::info!("Successfully added word_count column");
    } else {
        log::info!("word_count column already exists");
    }

    // Migration: Add status column to chapters if it doesn't exist
    if !column_exists(pool, "chapters", "status").await? {
        log::info!("Adding status column to chapters...");
        sqlx::query("ALTER TABLE chapters ADD COLUMN status TEXT DEFAULT 'draft'")
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to add status column: {}", e))?;
        log::info!("Successfully added status column");
    } else {
        log::info!("status column already exists");
    }

    log::info!("Database migrations completed successfully");
    Ok(())
}

// ============================================================================
// Project Operations
// ============================================================================

pub async fn get_projects() -> Result<Vec<ProjectWithCounts>> {
    let pool = get_pool()?;
    let rows = sqlx::query(
        r#"
        SELECT
            p.id, p.name, p.description, p.cover_image, p.genre,
            p.created_at, p.updated_at,
            (SELECT COUNT(*) FROM chapters WHERE project_id = p.id) as chapter_count,
            (SELECT COUNT(*) FROM characters WHERE project_id = p.id) as character_count,
            (SELECT COUNT(*) FROM locations WHERE project_id = p.id) as location_count,
            (SELECT COUNT(*) FROM lore_notes WHERE project_id = p.id) as lore_note_count
        FROM projects p
        ORDER BY p.updated_at DESC
        "#,
    )
    .fetch_all(&pool)
    .await?;

    let projects = rows
        .iter()
        .map(|row| ProjectWithCounts {
            id: row.get("id"),
            name: row.get("name"),
            description: row.get("description"),
            cover_image: row.get("cover_image"),
            genre: row.get("genre"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            chapter_count: row.get("chapter_count"),
            character_count: row.get("character_count"),
            location_count: row.get("location_count"),
            lore_note_count: row.get("lore_note_count"),
        })
        .collect();

    Ok(projects)
}

pub async fn get_project(id: &str) -> Result<Option<Project>> {
    let pool = get_pool().map_err(|e| anyhow::anyhow!("Failed to get database connection while fetching project '{}': {}", id, e))?;
    let row = sqlx::query(
        "SELECT id, name, description, cover_image, genre, created_at, updated_at FROM projects WHERE id = ?"
    )
    .bind(id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| anyhow::anyhow!("Database query failed while fetching project '{}': {}", id, e))?;

    Ok(row.map(|r| Project {
        id: r.get("id"),
        name: r.get("name"),
        description: r.get("description"),
        cover_image: r.get("cover_image"),
        genre: r.get("genre"),
        created_at: r.get("created_at"),
        updated_at: r.get("updated_at"),
    }))
}

pub async fn create_project(data: CreateProjectRequest) -> Result<Project> {
    let pool = get_pool()?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "INSERT INTO projects (id, name, description, genre, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)"
    )
    .bind(&id)
    .bind(&data.name)
    .bind(&data.description)
    .bind(&data.genre)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await?;

    // Create default project settings
    let settings_id = uuid::Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO project_settings (id, project_id) VALUES (?, ?)"
    )
    .bind(&settings_id)
    .bind(&id)
    .execute(&pool)
    .await?;

    Ok(Project {
        id,
        name: data.name,
        description: data.description,
        cover_image: None,
        genre: data.genre,
        created_at: now.clone(),
        updated_at: now,
    })
}

pub async fn update_project(id: &str, data: Project) -> Result<Project> {
    let pool = get_pool()?;
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "UPDATE projects SET name = ?, description = ?, cover_image = ?, genre = ?, updated_at = ? WHERE id = ?"
    )
    .bind(&data.name)
    .bind(&data.description)
    .bind(&data.cover_image)
    .bind(&data.genre)
    .bind(&now)
    .bind(id)
    .execute(&pool)
    .await?;

    Ok(Project {
        id: id.to_string(),
        name: data.name,
        description: data.description,
        cover_image: data.cover_image,
        genre: data.genre,
        created_at: data.created_at,
        updated_at: now,
    })
}

pub async fn delete_project(id: &str) -> Result<()> {
    let pool = get_pool()?;
    sqlx::query("DELETE FROM projects WHERE id = ?")
        .bind(id)
        .execute(&pool)
        .await?;
    Ok(())
}

// ============================================================================
// Chapter Operations
// ============================================================================

pub async fn get_chapters(project_id: &str) -> Result<Vec<Chapter>> {
    let pool = get_pool()?;
    let rows = sqlx::query(
        "SELECT id, project_id, title, content, order_index, word_count, status, created_at, updated_at FROM chapters WHERE project_id = ? ORDER BY order_index"
    )
    .bind(project_id)
    .fetch_all(&pool)
    .await?;

    let chapters = rows
        .iter()
        .map(|row| Chapter {
            id: row.get("id"),
            project_id: row.get("project_id"),
            title: row.get("title"),
            content: row.get("content"),
            order_index: row.get("order_index"),
            word_count: row.get("word_count"),
            status: row.get("status"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
        .collect();

    Ok(chapters)
}

pub async fn get_chapter(id: &str) -> Result<Option<Chapter>> {
    let pool = get_pool()?;
    let row = sqlx::query(
        "SELECT id, project_id, title, content, order_index, word_count, status, created_at, updated_at FROM chapters WHERE id = ?"
    )
    .bind(id)
    .fetch_optional(&pool)
    .await?;

    Ok(row.map(|r| Chapter {
        id: r.get("id"),
        project_id: r.get("project_id"),
        title: r.get("title"),
        content: r.get("content"),
        order_index: r.get("order_index"),
        word_count: r.get("word_count"),
        status: r.get("status"),
        created_at: r.get("created_at"),
        updated_at: r.get("updated_at"),
    }))
}

pub async fn create_chapter(data: CreateChapterRequest) -> Result<Chapter> {
    let pool = get_pool()?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    // Get max order
    let max_order: i32 = sqlx::query_scalar(
        "SELECT COALESCE(MAX(order_index), -1) FROM chapters WHERE project_id = ?"
    )
    .bind(&data.project_id)
    .fetch_one(&pool)
    .await?;

    let order = max_order + 1;
    let word_count = data.content.as_ref().map(|c| c.split_whitespace().count() as i32).unwrap_or(0);

    sqlx::query(
        "INSERT INTO chapters (id, project_id, title, content, order_index, word_count, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?)"
    )
    .bind(&id)
    .bind(&data.project_id)
    .bind(&data.title)
    .bind(&data.content)
    .bind(order)
    .bind(word_count)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await?;

    Ok(Chapter {
        id,
        project_id: data.project_id,
        title: data.title,
        content: data.content,
        order_index: order,
        word_count,
        status: "draft".to_string(),
        created_at: now.clone(),
        updated_at: now,
    })
}

pub async fn update_chapter(id: &str, data: Chapter) -> Result<Chapter> {
    let pool = get_pool()?;
    let now = chrono::Utc::now().to_rfc3339();
    let word_count = data.content.as_ref().map(|c| c.split_whitespace().count() as i32).unwrap_or(0);

    sqlx::query(
        "UPDATE chapters SET title = ?, content = ?, order_index = ?, word_count = ?, status = ?, updated_at = ? WHERE id = ?"
    )
    .bind(&data.title)
    .bind(&data.content)
    .bind(data.order_index)
    .bind(word_count)
    .bind(&data.status)
    .bind(&now)
    .bind(id)
    .execute(&pool)
    .await?;

    Ok(Chapter {
        id: id.to_string(),
        project_id: data.project_id,
        title: data.title,
        content: data.content,
        order_index: data.order_index,
        word_count,
        status: data.status,
        created_at: data.created_at,
        updated_at: now,
    })
}

pub async fn delete_chapter(id: &str) -> Result<()> {
    let pool = get_pool()?;
    sqlx::query("DELETE FROM chapters WHERE id = ?")
        .bind(id)
        .execute(&pool)
        .await?;
    Ok(())
}

// ============================================================================
// Character Operations
// ============================================================================

pub async fn get_characters(project_id: &str) -> Result<Vec<Character>> {
    let pool = get_pool()?;
    let rows = sqlx::query(
        "SELECT * FROM characters WHERE project_id = ? ORDER BY name"
    )
    .bind(project_id)
    .fetch_all(&pool)
    .await?;

    let characters = rows
        .iter()
        .map(|row| Character {
            id: row.get("id"),
            project_id: row.get("project_id"),
            name: row.get("name"),
            age: row.get("age"),
            gender: row.get("gender"),
            role: row.get("role"),
            occupation: row.get("occupation"),
            appearance: row.get("appearance"),
            distinguishing_features: row.get("distinguishing_features"),
            personality: row.get("personality"),
            traits: row.get("traits"),
            flaws: row.get("flaws"),
            fears: row.get("fears"),
            desires: row.get("desires"),
            background: row.get("background"),
            relationships: row.get("relationships"),
            skills: row.get("skills"),
            arc: row.get("arc"),
            motivation: row.get("motivation"),
            conflicts: row.get("conflicts"),
            speech_pattern: row.get("speech_pattern"),
            catchphrases: row.get("catchphrases"),
            notes: row.get("notes"),
            avatar: row.get("avatar"),
            color: row.get("color"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
        .collect();

    Ok(characters)
}

pub async fn create_character(data: CreateCharacterRequest) -> Result<Character> {
    let pool = get_pool()?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "INSERT INTO characters (id, project_id, name, age, gender, role, background, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&id)
    .bind(&data.project_id)
    .bind(&data.name)
    .bind(&data.age)
    .bind(&data.gender)
    .bind(&data.role)
    .bind(&data.background)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await?;

    Ok(Character {
        id,
        project_id: data.project_id,
        name: data.name,
        age: data.age,
        gender: data.gender,
        role: data.role,
        occupation: None,
        appearance: None,
        distinguishing_features: None,
        personality: None,
        traits: None,
        flaws: None,
        fears: None,
        desires: None,
        background: data.background,
        relationships: None,
        skills: None,
        arc: None,
        motivation: None,
        conflicts: None,
        speech_pattern: None,
        catchphrases: None,
        notes: None,
        avatar: None,
        color: None,
        created_at: now.clone(),
        updated_at: now,
    })
}

pub async fn update_character(id: &str, data: Character) -> Result<Character> {
    let pool = get_pool()?;
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        r#"UPDATE characters SET
            name = ?, age = ?, gender = ?, role = ?, occupation = ?,
            appearance = ?, distinguishing_features = ?, personality = ?,
            traits = ?, flaws = ?, fears = ?, desires = ?, background = ?,
            relationships = ?, skills = ?, arc = ?, motivation = ?, conflicts = ?,
            speech_pattern = ?, catchphrases = ?, notes = ?, avatar = ?, color = ?,
            updated_at = ?
        WHERE id = ?"#
    )
    .bind(&data.name)
    .bind(&data.age)
    .bind(&data.gender)
    .bind(&data.role)
    .bind(&data.occupation)
    .bind(&data.appearance)
    .bind(&data.distinguishing_features)
    .bind(&data.personality)
    .bind(&data.traits)
    .bind(&data.flaws)
    .bind(&data.fears)
    .bind(&data.desires)
    .bind(&data.background)
    .bind(&data.relationships)
    .bind(&data.skills)
    .bind(&data.arc)
    .bind(&data.motivation)
    .bind(&data.conflicts)
    .bind(&data.speech_pattern)
    .bind(&data.catchphrases)
    .bind(&data.notes)
    .bind(&data.avatar)
    .bind(&data.color)
    .bind(&now)
    .bind(id)
    .execute(&pool)
    .await?;

    Ok(Character {
        id: id.to_string(),
        project_id: data.project_id,
        name: data.name,
        age: data.age,
        gender: data.gender,
        role: data.role,
        occupation: data.occupation,
        appearance: data.appearance,
        distinguishing_features: data.distinguishing_features,
        personality: data.personality,
        traits: data.traits,
        flaws: data.flaws,
        fears: data.fears,
        desires: data.desires,
        background: data.background,
        relationships: data.relationships,
        skills: data.skills,
        arc: data.arc,
        motivation: data.motivation,
        conflicts: data.conflicts,
        speech_pattern: data.speech_pattern,
        catchphrases: data.catchphrases,
        notes: data.notes,
        avatar: data.avatar,
        color: data.color,
        created_at: data.created_at,
        updated_at: now,
    })
}

pub async fn delete_character(id: &str) -> Result<()> {
    let pool = get_pool()?;
    sqlx::query("DELETE FROM characters WHERE id = ?")
        .bind(id)
        .execute(&pool)
        .await?;
    Ok(())
}

// ============================================================================
// Location Operations
// ============================================================================

pub async fn get_locations(project_id: &str) -> Result<Vec<Location>> {
    let pool = get_pool()?;
    let rows = sqlx::query(
        "SELECT * FROM locations WHERE project_id = ? ORDER BY name"
    )
    .bind(project_id)
    .fetch_all(&pool)
    .await?;

    let locations = rows
        .iter()
        .map(|row| Location {
            id: row.get("id"),
            project_id: row.get("project_id"),
            name: row.get("name"),
            location_type: row.get("location_type"),
            description: row.get("description"),
            atmosphere: row.get("atmosphere"),
            features: row.get("features"),
            history: row.get("history"),
            notes: row.get("notes"),
            image: row.get("image"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
        .collect();

    Ok(locations)
}

pub async fn create_location(data: CreateLocationRequest) -> Result<Location> {
    let pool = get_pool()?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "INSERT INTO locations (id, project_id, name, location_type, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&id)
    .bind(&data.project_id)
    .bind(&data.name)
    .bind(&data.location_type)
    .bind(&data.description)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await?;

    Ok(Location {
        id,
        project_id: data.project_id,
        name: data.name,
        location_type: data.location_type,
        description: data.description,
        atmosphere: None,
        features: None,
        history: None,
        notes: None,
        image: None,
        created_at: now.clone(),
        updated_at: now,
    })
}

pub async fn update_location(id: &str, data: Location) -> Result<Location> {
    let pool = get_pool()?;
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        r#"UPDATE locations SET
            name = ?, location_type = ?, description = ?, atmosphere = ?,
            features = ?, history = ?, notes = ?, image = ?, updated_at = ?
        WHERE id = ?"#
    )
    .bind(&data.name)
    .bind(&data.location_type)
    .bind(&data.description)
    .bind(&data.atmosphere)
    .bind(&data.features)
    .bind(&data.history)
    .bind(&data.notes)
    .bind(&data.image)
    .bind(&now)
    .bind(id)
    .execute(&pool)
    .await?;

    Ok(Location {
        id: id.to_string(),
        project_id: data.project_id,
        name: data.name,
        location_type: data.location_type,
        description: data.description,
        atmosphere: data.atmosphere,
        features: data.features,
        history: data.history,
        notes: data.notes,
        image: data.image,
        created_at: data.created_at,
        updated_at: now,
    })
}

pub async fn delete_location(id: &str) -> Result<()> {
    let pool = get_pool()?;
    sqlx::query("DELETE FROM locations WHERE id = ?")
        .bind(id)
        .execute(&pool)
        .await?;
    Ok(())
}

// ============================================================================
// Lore Note Operations
// ============================================================================

pub async fn get_lore_notes(project_id: &str) -> Result<Vec<LoreNote>> {
    let pool = get_pool()?;
    let rows = sqlx::query(
        "SELECT * FROM lore_notes WHERE project_id = ? ORDER BY title"
    )
    .bind(project_id)
    .fetch_all(&pool)
    .await?;

    let lore_notes = rows
        .iter()
        .map(|row| LoreNote {
            id: row.get("id"),
            project_id: row.get("project_id"),
            title: row.get("title"),
            category: row.get("category"),
            content: row.get("content"),
            tags: row.get("tags"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
        .collect();

    Ok(lore_notes)
}

pub async fn create_lore_note(data: CreateLoreNoteRequest) -> Result<LoreNote> {
    let pool = get_pool()?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "INSERT INTO lore_notes (id, project_id, title, category, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&id)
    .bind(&data.project_id)
    .bind(&data.title)
    .bind(&data.category)
    .bind(&data.content)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await?;

    Ok(LoreNote {
        id,
        project_id: data.project_id,
        title: data.title,
        category: data.category,
        content: data.content,
        tags: None,
        created_at: now.clone(),
        updated_at: now,
    })
}

pub async fn update_lore_note(id: &str, data: LoreNote) -> Result<LoreNote> {
    let pool = get_pool()?;
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "UPDATE lore_notes SET title = ?, category = ?, content = ?, tags = ?, updated_at = ? WHERE id = ?"
    )
    .bind(&data.title)
    .bind(&data.category)
    .bind(&data.content)
    .bind(&data.tags)
    .bind(&now)
    .bind(id)
    .execute(&pool)
    .await?;

    Ok(LoreNote {
        id: id.to_string(),
        project_id: data.project_id,
        title: data.title,
        category: data.category,
        content: data.content,
        tags: data.tags,
        created_at: data.created_at,
        updated_at: now,
    })
}

pub async fn delete_lore_note(id: &str) -> Result<()> {
    let pool = get_pool()?;
    sqlx::query("DELETE FROM lore_notes WHERE id = ?")
        .bind(id)
        .execute(&pool)
        .await?;
    Ok(())
}

// ============================================================================
// Project Settings Operations
// ============================================================================

pub async fn get_project_settings(project_id: &str) -> Result<Option<ProjectSettings>> {
    let pool = get_pool()?;
    let row = sqlx::query(
        "SELECT * FROM project_settings WHERE project_id = ?"
    )
    .bind(project_id)
    .fetch_optional(&pool)
    .await?;

    Ok(row.map(|r| ProjectSettings {
        id: r.get("id"),
        project_id: r.get("project_id"),
        target_word_count: r.get("target_word_count"),
        daily_word_goal: r.get("daily_word_goal"),
        auto_save: r.get::<i32, _>("auto_save") == 1,
        auto_save_interval: r.get("auto_save_interval"),
        context_paragraphs: r.get("context_paragraphs"),
        genres: r.get("genres"),
        themes: r.get("themes"),
        target_audience: r.get("target_audience"),
        writing_style: r.get("writing_style"),
        narrative_pov: r.get("narrative_pov"),
        content_rating: r.get("content_rating"),
        content_warnings: r.get("content_warnings"),
        tone_preferences: r.get("tone_preferences"),
        time_period: r.get("time_period"),
        world_type: r.get("world_type"),
        language: r.get("language"),
        language_style: r.get("language_style"),
        adult_content: r.get("adult_content"),
        adult_intensity: r.get("adult_intensity"),
    }))
}

pub async fn update_project_settings(project_id: &str, data: ProjectSettings) -> Result<ProjectSettings> {
    let pool = get_pool()?;

    sqlx::query(
        r#"UPDATE project_settings SET
            target_word_count = ?, daily_word_goal = ?, auto_save = ?,
            auto_save_interval = ?, context_paragraphs = ?, genres = ?,
            themes = ?, target_audience = ?, writing_style = ?, narrative_pov = ?,
            content_rating = ?, content_warnings = ?, tone_preferences = ?,
            time_period = ?, world_type = ?, language = ?, language_style = ?,
            adult_content = ?, adult_intensity = ?
        WHERE project_id = ?"#
    )
    .bind(data.target_word_count)
    .bind(data.daily_word_goal)
    .bind(if data.auto_save { 1i32 } else { 0i32 })
    .bind(data.auto_save_interval)
    .bind(data.context_paragraphs)
    .bind(&data.genres)
    .bind(&data.themes)
    .bind(&data.target_audience)
    .bind(&data.writing_style)
    .bind(&data.narrative_pov)
    .bind(&data.content_rating)
    .bind(&data.content_warnings)
    .bind(&data.tone_preferences)
    .bind(&data.time_period)
    .bind(&data.world_type)
    .bind(&data.language)
    .bind(&data.language_style)
    .bind(&data.adult_content)
    .bind(&data.adult_intensity)
    .bind(project_id)
    .execute(&pool)
    .await?;

    Ok(data)
}

// ============================================================================
// Preset Operations
// ============================================================================

pub async fn get_presets() -> Result<Vec<GenerationPreset>> {
    let pool = get_pool()?;
    let rows = sqlx::query(
        "SELECT * FROM generation_presets ORDER BY name"
    )
    .fetch_all(&pool)
    .await?;

    let presets = rows
        .iter()
        .map(|row| GenerationPreset {
            id: row.get("id"),
            name: row.get("name"),
            preset_type: row.get("preset_type"),
            positive_prompt: row.get("positive_prompt"),
            negative_prompt: row.get("negative_prompt"),
            selected_tone: row.get("selected_tone"),
            custom_tone_instruction: row.get("custom_tone_instruction"),
            custom_style_name: row.get("custom_style_name"),
            custom_style_instruction: row.get("custom_style_instruction"),
            custom_genre_name: row.get("custom_genre_name"),
            custom_genre_instruction: row.get("custom_genre_instruction"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
        .collect();

    Ok(presets)
}

pub async fn create_preset(data: CreatePresetRequest) -> Result<GenerationPreset> {
    let pool = get_pool()?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        r#"INSERT INTO generation_presets
            (id, name, preset_type, positive_prompt, negative_prompt, selected_tone,
             custom_tone_instruction, custom_style_name, custom_style_instruction,
             custom_genre_name, custom_genre_instruction, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#
    )
    .bind(&id)
    .bind(&data.name)
    .bind(&data.preset_type)
    .bind(&data.positive_prompt)
    .bind(&data.negative_prompt)
    .bind(&data.selected_tone)
    .bind(&data.custom_tone_instruction)
    .bind(&data.custom_style_name)
    .bind(&data.custom_style_instruction)
    .bind(&data.custom_genre_name)
    .bind(&data.custom_genre_instruction)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await?;

    Ok(GenerationPreset {
        id,
        name: data.name,
        preset_type: data.preset_type,
        positive_prompt: data.positive_prompt,
        negative_prompt: data.negative_prompt,
        selected_tone: data.selected_tone,
        custom_tone_instruction: data.custom_tone_instruction,
        custom_style_name: data.custom_style_name,
        custom_style_instruction: data.custom_style_instruction,
        custom_genre_name: data.custom_genre_name,
        custom_genre_instruction: data.custom_genre_instruction,
        created_at: now.clone(),
        updated_at: now,
    })
}

pub async fn delete_preset(id: &str) -> Result<()> {
    let pool = get_pool()?;
    sqlx::query("DELETE FROM generation_presets WHERE id = ?")
        .bind(id)
        .execute(&pool)
        .await?;
    Ok(())
}
