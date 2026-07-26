-- List all tables in the current SQLite database
SELECT
  thread_id,
  GROUP_CONCAT(role || ': ' || content, ' /n/n ') AS conversation
FROM messages
GROUP BY thread_id;