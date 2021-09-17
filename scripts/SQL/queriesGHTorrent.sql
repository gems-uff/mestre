-- Queries used on the GHTorrent (ghtorrent.org) hosted on Google BigQuery (https://twitter.com/ghtorrent/status/1222529377629605889)
-- 1. Number of commits for a project
-- 2. Number of stars for a project


-- 1. Number of commits for a project
SELECT
  COUNT(*)
FROM
  ghtorrentmysql1906.MySQL1906.commits
WHERE
  project_id = (
  SELECT
    id
  FROM
    ghtorrentmysql1906.MySQL1906.projects
  WHERE
    owner_id = (
    SELECT
      id
    FROM
      ghtorrentmysql1906.MySQL1906.users
    WHERE
      login = 'android')
    AND name = 'platform_frameworks_base')
  AND created_at <= '2016-03-31'
  
  
  
-- 2. Number of stars for a project
  SELECT
  COUNT(repo_id)
FROM
  ghtorrentmysql1906.MySQL1906.watchers
WHERE
  repo_id = (
  SELECT
    id
  FROM
    ghtorrentmysql1906.MySQL1906.projects
  WHERE
    owner_id = (
    SELECT
      id
    FROM
      ghtorrentmysql1906.MySQL1906.users
    WHERE
      login = 'apache')
    AND name = 'lucene-solr')
  AND created_at <= '2016-03-31'
