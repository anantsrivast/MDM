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

def ngrams_min_hash(line,tuple_size,coeffs,r):
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
    sv=SparseVector(1296,dict(Counter(arr)))

    arr=[]
    '''
    Yield a sparse vector for each band
    '''
    for i in xrange(r):

        yield words[0],i,sv



def gen_hash_coeffs_1(num_buckets,num_bigrams,r,seed):
    '''
    Generate num_buckets hash functions
    '''
    coeff=[None]*r
    bands=num_buckets/r
    arr=[None]*bands
    for i in xrange(num_buckets):
        nRDD= RandomRDDs.uniformRDD(sc, num_bigrams,0,seed+i).map(lambda val: float(-1) if val <= 0.5 else float(1)).collect()
        j=nRDD
        idx= i%(bands)
        arr[idx]=(idx+1,j)
        if i%bands ==bands-1:
            coeff[i/bands]=arr
            arr=[None]*bands
    return coeff


def LSH(line,coeff):
    import numpy as np
    r=(coeff.value)[line[1]]
    hash_str=''
    for i in xrange(len(r)):
        ##improved- This is best
        '''
        return the sign of the dot product of the vector and the random hyperplane
        '''
        hash_str=hash_str+str(1 if np.array((line[2].toArray())).dot(np.array(r[i][1])).sum() > 0 else 0)
    return hash_str,(line[0],str(line[2]))

def load_source_data():
    pipeline='[{"$unwind": "$sources"},{"$project":{"_id":0,"id":"$sources._id","val":{"$toLower":{"$concat":["$sources.first_name","$sources.middle_name","$sources.last_name",{"$substr":["$sources.gender",0,1]},"$sources.dob","$sources.address.street","$sources.address.city","$sources.address.state","$sources.address.zip","$sources.phone","$sources.email"]}}}}]'

    source_df=spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri","mongodb://ec2-54-200-163-164.us-west-2.compute.amazonaws.com:27017/").option("database","single").option("collection","master").option("pipeline", pipeline).option("readPreference.name","secondaryPreferred").load()

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
            if dotp/rss > 0.75:
                if line[1][i][0] < line[1][j][0]:
                        yield line[1][i][0],line[1][j][0]
                else:
                        yield line[1][j][0],line[1][i][0]


def full_load_lsh(sc,spark,r,nbuckets,seed):

    num_rows=r
    num_buckets=nbuckets
    
    coeffs=sc.broadcast(gen_hash_coeffs_1(num_buckets,1296,num_rows,seed))
    source_df=load_source_data();
    source_rdd=source_df.rdd.map(tuple)
    doc_vector=source_rdd.flatMap(lambda line: ngrams_min_hash(line,2,coeffs,num_rows)).map(lambda line:LSH(line,coeffs)).combineByKey(lambda value:[value],lambda x,value:x+[value],lambda x,value:x+value).filter(lambda (x,y): len(y)>= 2)
    doc_vector.cache()
    '''
    call cosine similarity function here for each matching pair.
    This portion can be further optimized by deduplicating first
    '''
    x=doc_vector.flatMap(lambda line: cosine_pre_process(line)).reduceByKey(lambda x,y: x)
    write_df=spark.createDataFrame(x, ["f"])
    write_df.write.format("com.mongodb.spark.sql.DefaultSource").option("uri","mongodb://ec2-54-200-163-164.us-west-2.compute.amazonaws.com:27017/").option("database","sparkt").option("collection","initial_load").mode("append").save()


if __name__=="__main__":
    conf = SparkConf().setAppName("LSH")
    sc = SparkContext(conf=conf)
    spark=SQLContext(sc)
    full_load_lsh(sc,spark,40,800,500)
