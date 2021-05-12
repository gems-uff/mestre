-- Contents
-- 1.	INITIAL DATASET FOR MINING	1
-- 2.	NUMBER OF CONFLICTING CHUNKS PER PROJECT:	1
-- 3.	NUMBER OF TYPES OF DEVELOPER DECISION PER MERGE	2
-- 4.	NUMBER OF CONFLICTING CHUNKS PER MERGE	4
-- 5.	NUMBER OF DEVELOPER DECISION TYPES PER MERGE	4
-- 6.	NUMBER OF DEVELOPER DECISION TYPES PER CONFLICTING FILE	6



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
		r.project_id = p.id) as chunks
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
