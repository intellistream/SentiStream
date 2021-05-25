# -*- coding: utf-8 -*-

from pyflink.dataset import ExecutionEnvironment
from pyflink.table import TableConfig, DataTypes, BatchTableEnvironment
from pyflink.table.descriptors import Schema, OldCsv, FileSystem
from pyflink.table.udf import udf
import os

# input
source_file = 'tweets.text'
# output
sink_file = 'tweets_clean.csv'

# create environment
exec_env = ExecutionEnvironment.get_execution_environment()
exec_env.set_parallelism(1)
t_config = TableConfig()
t_env = BatchTableEnvironment.create(exec_env, t_config)

# delete output file if already exists
if os.path.exists(sink_file):
    os.remove(sink_file)

# create source
t_env.connect(FileSystem().path(source_file)) \
    .with_format(OldCsv()
                 .field_delimiter('\n')
                 .field('tweet', DataTypes.STRING())) \
    .with_schema(Schema()
                 .field('tweet', DataTypes.STRING())) \
    .create_temporary_table('mySource')

# create sink
t_env.connect(FileSystem().path(sink_file)) \
    .with_format(OldCsv()
                 .field('clean', DataTypes.STRING())) \
    .with_schema(Schema()
                 .field('clean', DataTypes.STRING())) \
    .create_temporary_table('mySink')

# udf Text preprocessing
@udf(input_types=[DataTypes.STRING()], result_type=DataTypes.STRING())
def textprocess(text):
    import re
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'^br$', ' ', text)
    text = re.sub(r'\s+^br$\s+', ' ', text)
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'^b\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text) 
    text = text.lower().strip()   
    return text
t_env.register_function('textprocess', textprocess) 


t_env.scan('mySource') \
    .select('textprocess(tweet) AS clean') \
    .insert_into('mySink')

# execute job
t_env.execute("textclean")
