-- Contents
-- 1.	INITIAL DATASET FOR MINING
-- 2.	NUMBER OF CONFLICTING CHUNKS PER PROJECT: (data/number_conflicting_chunks.csv)
-- 3.	NUMBER OF TYPES OF DEVELOPER DECISION PER MERGE
-- 4.	NUMBER OF CONFLICTING CHUNKS PER MERGE
-- 5.	NUMBER OF DEVELOPER DECISION TYPES PER MERGE (data/developerdecision_merge.csv)
-- 6.	NUMBER OF DEVELOPER DECISION TYPES PER CONFLICTING FILE (data/developerdecision_file.csv)
-- 7.	NUMBER OF MERGES PER PROJECT
-- 8.	NUMBER OF MERGES PER SELECTED PROJECT (data/number_merges_project_selected.csv)
-- 9	NUMBER OF CONFLICTING MERGES PER SELECTED PROJECT (data/number_conflicting_merges_project_selected.csv)


-- INITIAL DATASET FOR MINING  
-- 175805 CONFLICTING CHUNKS FROM 2731 PROJECTS

select 
	cc.id as chunk_id,
	cc.beginline,
	cc.endline,
	cc.developerdecision,
	cc.general_kind_conflict_outmost as kind_conflict,
	p.developers,
	p.searchurl,
	r.numberconflictingfiles,
	r.numberjavaconflictingfiles,
	cf.name as file_name,
r.sha,
	r.leftsha,
	r.rightsha,
	r.basesha
	
from conflictingchunk cc 
inner join conflictingfile cf on cc.conflictingfile_id = cf.id
inner join revision r on r.id = cf.revision_id
inner join project p on p.id = r.project_id
where cf.filetype='java' and 
p.fork = false and
p.analyzed = true




-- NUMBER OF CONFLICTING CHUNKS PER PROJECT:

select
	p.id,
	replace(p.htmlurl, 'https://github.com/', '') as project,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	inner join revision r on
		cf.revision_id = r.id
	where
		r.project_id = p.id and
		cf.filetype='java' and 
		p.fork = false and
		p.analyzed = true ) as chunks
from
	project p
where
	p.id in (
	select
		p2.id
	from
		conflictingchunk cc2
	inner join conflictingfile cf2 on
		cc2.conflictingfile_id = cf2.id
	inner join revision r2 on
		r2.id = cf2.revision_id
	inner join project p2 on
		p2.id = r2.project_id
	where
		cf2.filetype = 'java'
		and p2.fork = false
		and p2.analyzed = true)
order by
	chunks desc;



-- NUMBER OF TYPES OF DEVELOPER DECISION PER MERGE

select
	r.id as revision_id,
	r.sha as sha,
	replace(p.htmlurl, 'https://github.com/', '') as project,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id) as chunks,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Version 1') as version1,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Version 2') as version2,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Concatenation') as concatenation,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Combination') as combination,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Manual') as manual,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'None') as none
from
	revision r
inner join project p on
	r.project_id = p.id
where
	p.id in (
	select
		p2.id
	from
		conflictingchunk cc2
	inner join conflictingfile cf2 on
		cc2.conflictingfile_id = cf2.id
	inner join revision r2 on
		r2.id = cf2.revision_id
	inner join project p2 on
		p2.id = r2.project_id
	where
		cf2.filetype = 'java'
		and p2.fork = false
		and p2.analyzed = true)
order by
	chunks desc;

-- NUMBER OF CONFLICTING CHUNKS PER MERGE

select 
	r.id as revision_id, replace(p.htmlurl, 'https://github.com/', '') as project, r.sha as sha, count(cc.id) as chunks
from conflictingchunk cc 
inner join conflictingfile cf on cc.conflictingfile_id = cf.id 
inner join revision r on r.id = cf.revision_id 
inner join project p on p.id = r.project_id 
where cf.filetype='java' and 
p.fork = false and
p.analyzed = true
group by r.id, p.htmlurl 
order by chunks desc;




-- NUMBER OF DEVELOPER DECISION TYPES PER MERGE

select
	r.id as revision_id,
	r.sha as sha,
	replace(p.htmlurl, 'https://github.com/', '') as project,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id) as chunks,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Version 1') as version1,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Version 2') as version2,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Concatenation') as concatenation,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Combination') as combination,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'Manual') as manual,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	where
		cf.revision_id = r.id
		and cc.developerdecision = 'None') as none
from
	revision r
inner join project p on
	r.project_id = p.id
where
	r.id in (
	select
		distinct r.id
	from
		conflictingchunk cc
	inner join conflictingfile cf on
		cc.conflictingfile_id = cf.id
	inner join revision r on
		r.id = cf.revision_id
	inner join project p on
		p.id = r.project_id
	where
		cf.filetype = 'java'
		and p.fork = false
		and p.analyzed = true
		and r.status = 'CONFLICTING')
order by
	chunks desc;




-- NUMBER OF DEVELOPER DECISION TYPES PER CONFLICTING FILE

select
	cf.id as file_id,
	cf.name as file_name,
	cf."path" as file_path,
	r.sha as sha,
	replace(p.htmlurl, 'https://github.com/', '') as project,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	where
		cf2.id = cf.id) as chunks,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	where
		cf2.id = cf.id
		and cc.developerdecision = 'Version 1') as version1,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	where
		cf2.id = cf.id
		and cc.developerdecision = 'Version 2') as version2,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	where
		cf2.id = cf.id
		and cc.developerdecision = 'Concatenation') as concatenation,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	where
		cf2.id = cf.id
		and cc.developerdecision = 'Combination') as combination,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	where
		cf2.id = cf.id
		and cc.developerdecision = 'Manual') as MANUAL,
	(
	select
		count(cc.id)
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	where
		cf2.id = cf.id
		and cc.developerdecision = 'None') as none
from
	conflictingfile cf
inner join revision r on
	cf.revision_id = r.id
inner join project p on
	r.project_id = p.id
where
	cf.id in (
	select
		distinct cf2.id
	from
		conflictingchunk cc
	inner join conflictingfile cf2 on
		cc.conflictingfile_id = cf2.id
	inner join revision r on
		r.id = cf2.revision_id
	inner join project p on
		p.id = r.project_id
	where
		cf2.filetype = 'java'
		and p.fork = false
		and p.analyzed = true
		and r.status = 'CONFLICTING')
order by
	chunks desc;

-- NUMBER OF MERGES PER PROJECT
select 
	p.id,
	replace(p.htmlurl, 'https://github.com/', '') as project,
	count(r.id) as nr_merges
	
		
from revision r
inner join project p on p.id = r.project_id 
where
p.fork = false and
p.analyzed = true and 
p. id in (
select 
	distinct p.id
	
from conflictingchunk cc 
inner join conflictingfile cf on cc.conflictingfile_id = cf.id
inner join revision r on r.id = cf.revision_id
inner join project p on p.id = r.project_id
where cf.filetype='java' and 
p.fork = false and
p.analyzed = true
)
group by p.id;


-- ID OF SELECTED PROJECTS
(65885, 185026, 206437, 217482, 223355, 507775, 726492, 762119, 961036, 1006053, 1022930, 1775980, 1795594, 
1965842, 1971081, 2045207, 2138392, 2230984, 2524488, 2709026, 2902099, 3129899, 3405664, 3518171, 3518362, 
3661343, 4212733, 4310801, 50229487)

-- NUMBER OF MERGES PER SELECTED PROJECT
select 
	p.id,
	replace(p.htmlurl, 'https://github.com/', '') as project,
	count(r.id) as nr_merges
	
		
from revision r
inner join project p on p.id = r.project_id 
where
p.fork = false and
p.analyzed = true and 
p. id in (65885, 185026, 206437, 217482, 223355, 507775, 726492, 762119, 961036, 1006053, 1022930, 1775980, 1795594, 
1965842, 1971081, 2045207, 2138392, 2230984, 2524488, 2709026, 2902099, 3129899, 3405664, 3518171, 3518362, 
3661343, 4212733, 4310801, 50229487)
group by p.id;

-- NUMBER OF CONFLICTING MERGES PER SELECTED PROJECT
select 
	p.id,
	replace(p.htmlurl, 'https://github.com/', '') as project,
	count(r.id) as nr_conflicting_merges
	
		
from revision r
inner join project p on p.id = r.project_id 
where
p.fork = false and
p.analyzed = true and
r.status = 'CONFLICTING' and
p. id in (65885, 185026, 206437, 217482, 223355, 507775, 726492, 762119, 961036, 1006053, 1022930, 1775980, 1795594, 
1965842, 1971081, 2045207, 2138392, 2230984, 2524488, 2709026, 2902099, 3129899, 3405664, 3518171, 3518362, 
3661343, 4212733, 4310801, 50229487)
group by p.id;