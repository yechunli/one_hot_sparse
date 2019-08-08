import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import tensorflow as tf

student_file = 'F:\python_project\\one_hot_sparse\\student.csv'
teacher_file = 'F:\python_project\\one_hot_sparse\\teacher.csv'
test_file = 'F:\python_project\\one_hot_sparse\\test.csv'

student = pd.read_csv(student_file)
teacher = pd.read_csv(teacher_file)
test = pd.read_csv(test_file)

train = pd.concat([student, teacher])
data = pd.concat([train, test])
data = data.fillna(-1)

test = data[data.label == -1]
train = data[data.label != -1]
label = train.pop('label')
label = label.values.reshape(-1, 1)
print(train, label)
# print('data', data)
# print('test', test)
# print('train', train)
#print('label', label)

feature = ['age', 'gender', 'money']


enc = OneHotEncoder()

enc.fit(data['id'].values.reshape(-1, 1))
train_data = enc.transform(train['id'].values.reshape(-1, 1))
test_data = enc.transform(test['id'].values.reshape(-1, 1))

for item in feature:
    enc.fit(data[item].values.reshape(-1, 1))
    train_tmp = enc.transform(train[item].values.reshape(-1, 1))
    test_tmp = enc.transform(test[item].values.reshape(-1, 1))

    train_data = sparse.hstack([train_data, train_tmp])
    test_data = sparse.hstack([test_data, test_tmp])

train_data = train_data.toarray()
test_data = test_data.toarray()

low = np.shape(train_data)[1]

train_x = tf.placeholder(tf.float32, shape=[None, low])
train_y = tf.placeholder(tf.float32, shape=[None, 1])

def parameter_init(row, low):
    weight = tf.Variable(tf.truncated_normal([row, low], stddev=0.1))
    bias = tf.Variable(tf.constant(0, shape=[1, low], dtype=tf.float32))
    return weight, bias

weight1, bias1 = parameter_init(low, 50)
weight2, bias2 = parameter_init(50, 1)

mul1 = tf.matmul(train_x, weight1)
add1 = tf.add(mul1, bias1)
act1 = tf.nn.tanh(add1)

mul2 = tf.matmul(act1, weight2)
add2 = tf.add(mul2, bias2)
act2 = tf.nn.sigmoid(add2)

loss = tf.reduce_mean((train_y - act2) ** 2)
auc_value = tf.metrics.auc(train_y, act2)

with tf.name_scope('evaluation'):
    tf.summary.scalar('loss', loss)
    #tf.summary.scalar('auc', auc_value[0])
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)


# init_global = tf.global_variables_initializer()
# init_local = tf.local_variables_initializer()
init_global = tf.initialize_all_variables()
init_local = tf.initialize_local_variables()
merge = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run([init_global, init_local])
    writer = tf.summary.FileWriter('F:\python_project\log\one_hot', graph=sess.graph)
    for i in range(100):
        summary, _, loss1, auc, act, y = sess.run([merge, train_op, loss, auc_value, act2, train_y], feed_dict={train_x:train_data, train_y:label})
        #_, loss1, act, y = sess.run([train_op, loss, act2, train_y], feed_dict={train_x: train_data, train_y: label})
        writer.add_summary(summary, i)
        #print('act=', act.reshape(1, -1))
        #print('y=', y.reshape(1, -1))
    saver = tf.train.Saver()
    saver.save(sess, 'F:\data')

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'F:\data')
    accuracy = sess.run(act2, feed_dict={train_x:test_data})
    print('loss= {}, auc= {}'.format(loss1, auc))
    print('test {}' .format(accuracy))



