-- ============================================
-- 越群山知识库RAG系统 — 数据库初始化脚本
-- 数据库：ox | 模式：root
-- ============================================

-- 创建 schema
CREATE SCHEMA IF NOT EXISTS root;

-- 1. 知识库表
CREATE TABLE IF NOT EXISTS root.knowledge_bases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. 原始文档表
CREATE TABLE IF NOT EXISTS root.documents (
    id SERIAL PRIMARY KEY,
    kb_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    file_size BIGINT NOT NULL,
    page_count INT DEFAULT 0,
    upload_status VARCHAR(20) DEFAULT 'pending',
    vector_status VARCHAR(20) DEFAULT 'pending',
    chunk_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (kb_id) REFERENCES root.knowledge_bases(id) ON DELETE CASCADE
);

-- 3. 文档分块表
CREATE TABLE IF NOT EXISTS root.document_chunks (
    id SERIAL PRIMARY KEY,
    kb_id INT NOT NULL,
    doc_id INT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INT NOT NULL,
    page_num INT DEFAULT 0,
    start_pos INT DEFAULT 0,
    end_pos INT DEFAULT 0,
    milvus_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (doc_id) REFERENCES root.documents(id) ON DELETE CASCADE
);

-- 4. 分块配置表
CREATE TABLE IF NOT EXISTS root.chunk_configs (
    id SERIAL PRIMARY KEY,
    kb_id INT UNIQUE NOT NULL,
    chunk_size INT DEFAULT 1000,
    chunk_overlap INT DEFAULT 100,
    split_separator TEXT DEFAULT '\n\n',
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (kb_id) REFERENCES root.knowledge_bases(id) ON DELETE CASCADE
);

-- 5. 对话历史表
CREATE TABLE IF NOT EXISTS root.chat_histories (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    kb_id INT NOT NULL,
    model_name VARCHAR(50) DEFAULT 'qwen3-max',
    user_query TEXT NOT NULL,
    ai_answer TEXT NOT NULL,
    reference_chunks JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_histories_session_id ON root.chat_histories(session_id);

-- 6. 系统配置表
CREATE TABLE IF NOT EXISTS root.system_configs (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(50) NOT NULL UNIQUE,
    config_value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- RAG评测模块 — 5张专用表（rag_eval_）
-- ============================================

-- 7. 评测数据集主表
CREATE TABLE IF NOT EXISTS root.rag_eval_datasets (
    id SERIAL PRIMARY KEY,
    kb_id INT NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    total_questions INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_rag_eval_datasets_kb_id ON root.rag_eval_datasets(kb_id);

-- 8. 评测问题表
CREATE TABLE IF NOT EXISTS root.rag_eval_questions (
    id SERIAL PRIMARY KEY,
    dataset_id INT NOT NULL,
    kb_id INT NOT NULL,
    query TEXT NOT NULL,
    standard_answer TEXT,
    standard_doc_ids INT[],
    standard_chunk_ids INT[],
    difficulty VARCHAR(20) DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (dataset_id) REFERENCES root.rag_eval_datasets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rag_eval_questions_dataset_id ON root.rag_eval_questions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_rag_eval_questions_kb_id ON root.rag_eval_questions(kb_id);

-- 9. 评测任务表
CREATE TABLE IF NOT EXISTS root.rag_eval_tasks (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL DEFAULT '',
    dataset_id INT NOT NULL,
    kb_id INT NOT NULL,
    top_k INT DEFAULT 5,
    retriever_mode VARCHAR(20) DEFAULT 'normal',
    model_name VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    progress INT DEFAULT 0,
    recall REAL,
    precision REAL,
    hit_rate REAL,
    mrr REAL,
    cost_seconds REAL,
    created_at TIMESTAMP DEFAULT NOW(),
    finished_at TIMESTAMP NULL,
    FOREIGN KEY (dataset_id) REFERENCES root.rag_eval_datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (kb_id) REFERENCES root.knowledge_bases(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rag_eval_tasks_dataset_id ON root.rag_eval_tasks(dataset_id);
CREATE INDEX IF NOT EXISTS idx_rag_eval_tasks_status ON root.rag_eval_tasks(status);

-- 10. 评测结果明细表
CREATE TABLE IF NOT EXISTS root.rag_eval_results (
    id SERIAL PRIMARY KEY,
    task_id INT NOT NULL,
    qid INT NOT NULL,
    query TEXT NOT NULL,
    retrieved_chunk_ids INT[],
    retrieved_doc_ids INT[],
    recall REAL,
    precision REAL,
    hit BOOLEAN,
    rank INT,
    mrr REAL,
    retrieve_time REAL,
    answer_time REAL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (task_id) REFERENCES root.rag_eval_tasks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rag_eval_results_task_id ON root.rag_eval_results(task_id);
CREATE INDEX IF NOT EXISTS idx_rag_eval_results_qid ON root.rag_eval_results(qid);

-- 11. 标注任务表
CREATE TABLE IF NOT EXISTS root.rag_eval_label_tasks (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    kb_id INT NOT NULL,
    top_k INT DEFAULT 5,
    description TEXT,
    created_by VARCHAR(100),
    status VARCHAR(20) DEFAULT 'in_progress',
    progress INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rag_eval_label_tasks_kb_id ON root.rag_eval_label_tasks(kb_id);
CREATE INDEX IF NOT EXISTS idx_rag_eval_label_tasks_status ON root.rag_eval_label_tasks(status);

-- 12. 标注详情表
CREATE TABLE IF NOT EXISTS root.rag_eval_label_details (
    id SERIAL PRIMARY KEY,
    task_id INT NOT NULL,
    query TEXT NOT NULL,
    standard_answer TEXT,
    standard_chunk_ids INT[],
    standard_doc_ids INT[],
    status VARCHAR(20) DEFAULT 'unannotated',
    annotated_by VARCHAR(100),
    annotated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (task_id) REFERENCES root.rag_eval_label_tasks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rag_eval_label_details_task_id ON root.rag_eval_label_details(task_id);
CREATE INDEX IF NOT EXISTS idx_rag_eval_label_details_status ON root.rag_eval_label_details(status);

-- ============================================
-- RAGAS评测扩展字段（给 rag_eval_tasks 和 rag_eval_results 加 RAGAS 指标列）
-- 每行一条 ALTER TABLE，不使用 DO 双美元块（避免 init_db 按分号拆分时破坏语句）
-- ============================================
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS enable_ragas BOOLEAN DEFAULT FALSE;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS eval_model VARCHAR(50);
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS context_precision REAL;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS context_recall REAL;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS faithfulness REAL;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS answer_relevancy REAL;

-- 聊天抽样评测扩展字段
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS task_type VARCHAR(20) DEFAULT 'manual';
ALTER TABLE root.rag_eval_tasks ALTER COLUMN dataset_id DROP NOT NULL;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS sample_time_start TIMESTAMP;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS sample_time_end TIMESTAMP;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS sample_count INT DEFAULT 10;
ALTER TABLE root.rag_eval_tasks ADD COLUMN IF NOT EXISTS sample_strategy VARCHAR(20) DEFAULT 'random';

ALTER TABLE root.rag_eval_results ADD COLUMN IF NOT EXISTS answer TEXT;
ALTER TABLE root.rag_eval_results ADD COLUMN IF NOT EXISTS context_precision REAL;
ALTER TABLE root.rag_eval_results ADD COLUMN IF NOT EXISTS context_recall REAL;
ALTER TABLE root.rag_eval_results ADD COLUMN IF NOT EXISTS faithfulness REAL;
ALTER TABLE root.rag_eval_results ADD COLUMN IF NOT EXISTS answer_relevancy REAL;

-- ============================================
-- RAG监控模块 — 2张埋点表
-- ============================================

-- 13. 对话埋点主表
CREATE TABLE IF NOT EXISTS root.chat_traces (
    id BIGSERIAL PRIMARY KEY,
    chat_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64),
    kb_id INT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT,
    retrieved_chunk_ids TEXT,
    used_chunk_ids TEXT,
    prompt_tokens INT DEFAULT 0,
    completion_tokens INT DEFAULT 0,
    total_tokens INT DEFAULT 0,
    search_cost_ms INT DEFAULT 0,
    llm_cost_ms INT DEFAULT 0,
    total_cost_ms INT DEFAULT 0,
    llm_model VARCHAR(64),
    feedback VARCHAR(32),
    status VARCHAR(32) DEFAULT 'success',
    create_time TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_traces_chat_id ON root.chat_traces(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_traces_create_time ON root.chat_traces(create_time);
CREATE INDEX IF NOT EXISTS idx_chat_traces_kb_id ON root.chat_traces(kb_id);

-- 14. 检索分块明细表
CREATE TABLE IF NOT EXISTS root.chat_chunk_details (
    id BIGSERIAL PRIMARY KEY,
    chat_id VARCHAR(64) NOT NULL,
    chunk_id INT NOT NULL,
    similarity_score REAL,
    rank_num INT,
    is_used INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_chunk_details_chat_id ON root.chat_chunk_details(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_chunk_details_chunk_id ON root.chat_chunk_details(chunk_id);

-- ============================================
-- 客户端访问记录表 — 审计/统计
-- 按 IP/会话首次去重写入（30 分钟窗口），详见 services/access_log_service.py
-- ============================================

-- 15. 访问记录表
CREATE TABLE IF NOT EXISTS root.access_log (
    id          BIGSERIAL PRIMARY KEY,
    client_ip   VARCHAR(64),
    session_id  VARCHAR(64),
    user_id     VARCHAR(64),
    method      VARCHAR(10),
    path        VARCHAR(500),
    status_code INT,
    user_agent  TEXT,
    referer     VARCHAR(500),
    cost_ms     BIGINT,
    create_time TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_access_log_ip_time ON root.access_log(client_ip, create_time DESC);
CREATE INDEX IF NOT EXISTS idx_access_log_time ON root.access_log(create_time DESC);

-- ============================================
-- 字段扩展：system_prompt / user_prompt
-- ============================================
ALTER TABLE root.chat_traces ADD COLUMN IF NOT EXISTS system_prompt TEXT;
ALTER TABLE root.chat_traces ADD COLUMN IF NOT EXISTS user_prompt TEXT;
