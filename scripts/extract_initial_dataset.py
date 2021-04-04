import database
import pandas as pd

query = """select 
            cc.id as chunk_id,
            cc.developerdecision,
            cc.general_kind_conflict_outmost as kind_conflict,
            p.htmlurl as url,
            replace(p.htmlurl, 'https://github.com/', '') as project,  
            split_part(replace(p.htmlurl, 'https://github.com/', ''), '/', 1) as project_user, 
            split_part(replace(p.htmlurl, 'https://github.com/', ''), '/', 2) as project_name,
            concat(split_part(replace(p.htmlurl, 'https://github.com/', ''), '/', 1), '/' , substring(replace(cf.path, '/home/gleiph/Desktop/analysis/rep', ''),3, char_length(replace(cf.path, '/home/gleiph/Desktop/analysis/rep', '')))) AS path,
            cf.name as file_name,
            r.sha,
            r.leftsha,
            r.rightsha,
            r.basesha
        from 
            conflictingchunk cc 
        inner join 
            conflictingfile cf on cc.conflictingfile_id = cf.id
        inner join 
            revision r on r.id = cf.revision_id
        inner join 
            project p on p.id = r.project_id
        where 
            cf.filetype='java' and 
            p.fork = false and
            p.analyzed = true
       """

def main():
    conn, cur = database.connect()
    dat = pd.read_sql_query(query, conn)
    print(dat.head())

    dat.to_csv('../data/INITIAL_DATASET.csv', index=None)

main()