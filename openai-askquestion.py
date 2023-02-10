import os
import openai
import pandas as pd
import numpy as np
import docx
import datetime
from openpyxl import Workbook

def get_test_sample(model=None,length=[],temperature=[],question=0):
    #test is the input parameters have been properly added
    ls=get_test_prompt()
    if model==None or length==[] or question not in np.array(ls).T[1] or len(length)!=len(temperature) or max(temperature)>1 or min(temperature)<0:
        return print('Request cannot be performed, wrong argument')
    #generate the answers
    AI_genarated_answers=generate_AI_genrated_prompt(model,length,temperature,ls[question-1][2])
    #save the answers in am excel file
    save_AI_answers(AI_genarated_answers,question)

def save_AI_answers(answers):
    wb=Workbook()
    ws=wb.active
    ws.cell(row=1, column=1, value='AI answer')
    ws.cell(row=1, column=2, value='Model')
    ws.cell(row=1, column=3, value='Temperature')
    for i in range (len(answers)):
        for j in range(len(answers[0])):
            ws.cell(row=2+i,column=j+1,value=answers[i][j])
    date=datetime.datetime.today().strftime('%Y-%m-%d')
    os.chdir(r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\AI generate')
    wb.save(f'AI answers question {question} using the model {answers[0][1]} {date}.xlsx')

    
def generate_AI_genrated_prompt(model,length,temperature, question):
    answers=[]
    #concatenate the prompt
    quest=concatenate_question(question)
    #call open AI IPA
    for i in range(len(length)):
        answers.append(open_ai_API(model,quest,length[i],temperature[i]),model,temperature)
    return answers

def open_ai_API(model,prompt,ln,temp):
    return openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temp,
            max_tokens=ln,
            n=1,)
def concatenate_question(ls):
    return '\n'.join(ls)

def get_test_prompt():
    #get the  questions from the .docx files
    path_report=r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\1\Essay_Set_Descriptions\Essay_Set_Descriptions'
    list=os.listdir(path_report)#list the docx files
    ls=[]
    #create a list of the docx used for the datasets
    for idx, doc in enumerate(list):
        if doc[:5]=='Essay':
            ls.append([doc,int(doc[doc.find('#')+1])])
    os.chdir(path_report)
    #get the prompt that was used on the student
    for idx, i in enumerate(ls):
        f=open(i[0],'rb')
        doc=docx.Document(f)
        start_source_essay, start_prompt, end_prompt=None, None, None
        for id, j in enumerate(doc.paragraphs):
            try:
                if j.runs[0].text=='Source Essay':
                    start_source_essay=id
                elif j.runs[0].text=='Prompt':
                    start_prompt=id
                elif j.runs[0].text[:17]=='Rubric Guidelines':
                    end_prompt=id
                    break
            except:
                continue
        prompt=[]
        if start_source_essay==None:
            beg=start_prompt
        else:
            beg=start_source_essay
        for j in range(beg+1,end_prompt):
            try:
                prompt.append(doc.paragraphs[j].runs[0].text)
            except:
                continue
        ls[idx].append(prompt)
    return ls
    

get_test_sample(model='text-ada-001', length=[150], temperature=[0.4], question=5)
#pd.read_excel('C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\1\training_set_rel3.xls')