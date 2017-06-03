import binascii
import collections
import re
import numpy as np
from collections import Counter
from pyspark.mllib.linalg import SparseVector
from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext
from  pyspark.mllib.random import RandomRDDs

def bigram_map():
    '''
    A 1296 dimension map
    '''
    a=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    c=0
    d=collections.OrderedDict()
    for i in a:
        for j in a:

            d[i+j]=c
            c+=1
    return d

def ngrams_min_hash(line,tuple_size,r,coeff):
    '''
    Formatting input
    '''
    line_new=','.join(x if x!= None else '' for x in line )
    line_new=''.join(line_new)

    words=line_new.split(",",1)
    arr=[]
    dict_bigram=bigram_map()

    name_temp=re.sub(r'\s+', '', words[1])
    name=re.sub("[^\w]",'',name_temp).lower()
    '''
    Create Bigrams
    '''
    for i in xrange((len(name)-tuple_size+1)):
        j=1
        tempGram=""
        while(j<tuple_size):
            tempGram=tempGram+name[i+j]
            j+=1
        
        arr.append(dict_bigram[name[i]+tempGram])
    '''
    Sparse Vector
    '''
    #sv=SparseVector(1296,dict(Counter(arr)))

    xv=SparseVector(1296,dict(Counter(arr)))
    sv=xv.toArray()
    i=0
    coeff1=coeff.value
    hash_arr=[]  
    hash_str=""
    hash_str1=""
    bnd=0
    while(i< len(coeff1)):
      hash_str=hash_str+str(1 if np.dot((coeff1[i]),sv) > 0 else 0)
      i+=1
      if(i%r==0):
        bnd+=1
        hash_str1=hash_str
        hash_str=""
        yield hash_str1+str(bnd),(words[0],str(xv))



def gen_hash_coeffs_1(num_buckets,num_bigrams,seed):
    coeff=[0 for j in xrange(num_buckets)]
    for i in xrange(num_buckets):
        #nRDD= RandomRDDs.normalRDD(sc, num_bigrams, seed=seed+i).map(lambda val: str(-1) if val < 0 else str(1)).collect()
        #nRDD= RandomRDDs.uniformRDD(sc, num_bigrams,0,seed+i).map(lambda val: str(-1) if val <= 0.5 else str(1)).collect()
        #Str to float
        nRDD= RandomRDDs.uniformRDD(sc, num_bigrams,0,seed+i).map(lambda val: float(-1) if val <= 0.5 else float(1)).collect()
        j=nRDD
        coeff[i]=j
    return coeff


def load_source_data():
    pipeline='[{"$unwind": "$sources"},{"$project":{"_id":0,"id":"$sources._id","val":{"$toLower":{"$concat":["$sources.first_name","$sources.middle_name","$sources.last_name",{"$substr":["$sources.gender",0,1]},"$sources.dob","$sources.address.street","$sources.address.city","$sources.address.state","$sources.address.zip","$sources.phone","$sources.email"]}}}}]'

    source_df=spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri","mongodb://34.223.248.161:27017/").option("database","Single").option("collection","master").option("pipeline", pipeline).option("partitioner", "MongoSamplePartitioner").option("partitionSizeMB","1").load()

    return source_df

def cosine_pre_process(line):
    length_matches=len(line[1])
    i=0
    j=0
    s1=SparseVector(1,[0],[1])
    s2=SparseVector(1,[0],[1])
    for i in xrange(length_matches-1):
        j=i

        while(j<length_matches-1):
            j=j+1
            sf=s1.parse(line[1][i][1])
            ss=(s2.parse(line[1][j][1]))
            dotp=sf.dot(ss)
            rss=np.sqrt(sum(np.square(sf.values)))*np.sqrt(sum(np.square(ss.values)))
            if dotp/rss > .60:
                
                if line[1][i][0] < line[1][j][0]:
                        yield line[1][i][0],line[1][j][0]
                else:
                        yield line[1][j][0],line[1][i][0]


def full_load_lsh(sc,spark,nbands,nbuckets,seed,outname):

    num_bands=nbands
    num_buckets=nbuckets
     #num_bands=5
     #num_buckets=10
    ##seed=1000
    coeffs=sc.broadcast(gen_hash_coeffs_1(num_buckets,1296,500))
    source_df=load_source_data();
     #new_df=source_df.drop("_id")
    #source_rdd=source_df.rdd.map(tuple)
    source_rdd_1=source_df.rdd.map(tuple)
     #source_rdd.partitionBy()
    source_rdd=source_rdd_1.repartition(80)
     ###Latest
    
    
    #doc_vector1=doc_vector2.repartition(320)
    
    doc_vector=source_rdd.flatMap(lambda line: ngrams_min_hash(line,2,num_bands,coeffs)).combineByKey(lambda value:[value],lambda x,value:x+[value],lambda x,value:x+value).filter(lambda (x,y): len(y)>= 2)
    
    doc_vector.cache()
     ##call cosine similarity function here
    #x=doc_vector.flatMap(lambda line: cosine_pre_process(line)).reduceByKey(lambda x,y: x)
    x=doc_vector.flatMap(lambda line: cosine_pre_process(line)).distinct().combineByKey(lambda value:[value],lambda x,value:x+[value],lambda x,value:x+value)
    #print(x.take(100))
    #x=doc_vector.flatMap(lambda line: cosine_pre_process_new(line)).count()
    write_df=spark.createDataFrame(x, ["f"])
    write_df.write.format("com.mongodb.spark.sql.DefaultSource").option("uri","mongodb://34.223.248.161:27017/").option("database","sparkc2").option("collection",outname).mode("append").save()
