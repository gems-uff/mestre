-- quantidade de 'arquivos com conflitos' por projeto

SELECT project.name, SUM(revision.numberconflictingfiles)
FROM project
INNER JOIN revision ON project.id = revision.project_id
GROUP BY project.name;


SELECT COUNT(project.name),
       SUM(revision.numberconflictingfiles) AS conflicting_files_on_project
FROM project
INNER JOIN revision ON project.id = revision.project_id;
-- count  | conflicting_files_on_project
-- ---------+------------------------------
--  1769705 |                       415206
-- (1 row)


SELECT COUNT(project.name) AS numero_de_projetos, 
       SUM(revision.numberconflictingfiles) AS conflicting_files_on_project
FROM project
INNER JOIN revision ON project.id = revision.project_id
GROUP BY numero_de_projetos, conflicting_files_on_project
HAVING conflicting_files_on_project > 0
-- GROUP BY project.name;


SELECT project.name, 
       SUM(revision.numberconflictingfiles) AS conflicting_files_on_project
FROM project
     INNER JOIN revision ON project.id = revision.project_id
GROUP BY project.name
HAVING conflicting_files_on_project > 0;


-- funcionou!
-- projetos com numero de conflitos > 0
SELECT project.name, 
       SUM(revision.numberconflictingfiles)
FROM project 
     INNER JOIN revision ON project.id = revision.project_id
GROUP BY project.name
HAVING SUM(revision.numberconflictingfiles) > 0;

--                         name                         |  sum
-- -----------------------------------------------------+-------
--  02343-CDIO-projekt                                  |    18
--  2                                                   |     6
--  2012-code                                           |    23
-- ...
-- (3822 rows)



-- [x] funcionou?
-- projetos com numero de conflitos em arquivos Java > 0
SELECT project.name AS project_name, 
       SUM(revision.numberjavaconflictingfiles) AS number_java_conflicting_files
FROM project 
     INNER JOIN revision ON project.id = revision.project_id
GROUP BY project.name
HAVING SUM(revision.numberjavaconflictingfiles) > 0
ORDER BY SUM(revision.numberjavaconflictingfiles) DESC
LIMIT 10;

-- (2963 rows)


-- teste pra filtrar os projetos que sao forks de outros projetos do dataset
-- por enquanto consegui apenas filtrar os forks, falta especificar que sao forks de projetos que estao no dataset!
SELECT project.name AS project_name, 
       SUM(revision.numberjavaconflictingfiles) AS number_java_conflicting_files
FROM project 
     INNER JOIN revision ON project.id = revision.project_id
WHERE project.fork = false
GROUP BY project.name
HAVING SUM(revision.numberjavaconflictingfiles) > 0
ORDER BY SUM(revision.numberjavaconflictingfiles) DESC;





SELECT *
FROM project
LIMIT 3;



SELECT COUNT(project.name), 
       SUM(revision.numberconflictingfiles)
FROM project
     INNER JOIN revision ON project.id = revision.project_id
HAVING SUM(revision.numberconflictingfiles) > 0;
--   count  |  sum
-- ---------+--------
--  1769705 | 415206
-- (1 row)


SELECT COUNT(project.name), 
       SUM(revision.numberconflictingfiles)
FROM revision
     INNER JOIN project ON project.id = revision.project_id
HAVING SUM(revision.numberconflictingfiles) > 0;



SELECT COUNT(project.name),
       SUM(revision.numberconflictingfiles)
FROM revision
     INNER JOIN project ON project.id = revision.project_id
GROUP BY revision.numberconflictingfiles 
HAVING revision.numberconflictingfiles > 0;

