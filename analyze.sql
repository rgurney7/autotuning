-- List all tables in the current SQLite database
SELECT
  thread_id,
  GROUP_CONCAT(role || ': ' || content, char(10) || char(10)) AS conversation
FROM messages
GROUP BY thread_id;