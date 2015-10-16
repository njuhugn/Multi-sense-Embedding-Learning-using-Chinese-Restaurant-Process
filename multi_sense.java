import java.util.*;
import java.util.Vector;
import java.io.*;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.math.BigDecimal;  
import java.math.RoundingMode;


public class multi_sense{
    static int word_window;//window_size for sentence
    static int dimension;
    static ArrayList<String>WORD=new ArrayList<String>();// word list
    static double[]prob_word;//appearing probablity for each word, used for word omitting 
    static double Alpha=0.025;//initial learning rate
    static double alpha=0.025;//current learning rate
    static math my_math=new math();
    static double[][] vect;//global word_embedding
    static double[][] vect_t;//embedding for nodes in haffman tree
    static ArrayList<vocab_word>vocab=new ArrayList<vocab_word>();//vocabulary
    static Random r = new Random();
    static HashMap<Integer,HashMap<Integer,double[]>>sense_match=new HashMap<Integer,HashMap<Integer,double[]>>();
    //sense sets for each token. each token is assocaited with a list of senses, each sense is assocaited with a embedding
    static HashMap<Integer,HashMap<Integer,Integer>>cusomers_in_table=new HashMap<Integer,HashMap<Integer,Integer>>();
    //Chinese restaurant process. each sense is assocaited with an integer indicating how many times words have been assigned
    static double gamma=0.1;// hyperparameter for Chinese restaurant process
    static String train_file="";

    public static void main(String[] args)throws Exception{
        String save_file="";
        String frequency_file="";
        for(int i=0;i<args.length;i++){
            if(args[i].equals("-dimension"))
                dimension=Integer.parseInt(args[i+1]); //dimensionality 
            if(args[i].equals("-word_window"))
                word_window=Integer.parseInt(args[i+1]);//context window
            if(args[i].equals("-save_file"))
                save_file=args[i+1];//saving file
            if(args[i].equals("-frequency_file"))
                frequency_file=args[i+1];//word frequency file
            if(args[i].equals("-train_file"))
                train_file=args[i+1];//training corpus
        }
        ReadFre(frequency_file);// read word frequency f
        vect=new double[prob_word.length][dimension];// global word embeddings
        vect_t=new double[prob_word.length-1][dimension];
        int n_doc=num_of_docs(train_file);
        //int n_doc=2100000;
        //ndoc=
        System.out.println("number of documents");
        System.out.println(n_doc);
        random();//Initialization
        binary_tree();//generating haffman tre
        int batch_size=50;//batch size
        int counter=0;//counter, counting how many documents have been processed
        int Iter=5;//total number of iterations to run
        int thread_num=16;// multi thread
        ExecutorService executor = Executors.newFixedThreadPool(thread_num);
        //Each document is comprised of a series of sentences and a sentence is comprised of a series of words
        for(int iter=0;iter<Iter;iter++){
            System.out.println("iteration "+iter);
            ArrayList<Future<ArrayList<int[]>>>list=new ArrayList<Future<ArrayList<int[]>>>();
            //return list for multi thread
            ArrayList<ArrayList<ArrayList<Integer>>>DocList=new ArrayList<ArrayList<ArrayList<Integer>>>();
            //List of Document for parallel processing
            ArrayList<ArrayList<Integer>>Doc=new ArrayList<ArrayList<Integer>>();;
            //current document
            ArrayList<ArrayList<int[]>>DocSenseList=new ArrayList<ArrayList<int[]>>();
            //When we do Chinese Restaurant Process, we need to keep track of label assignments from last iteration
            ArrayList<int[]>DocSense=new ArrayList<int[]>();
            //sense assignments for the current document from last iteration
            long start_time=System.currentTimeMillis();
            BufferedReader in=new BufferedReader(new FileReader(train_file));
            BufferedReader in_sense=null;
            if(iter!=0)in_sense=new BufferedReader(new FileReader("store_sense"+Integer.toString(iter-1)));
            FileWriter fw_new_sense=new FileWriter("store_sense"+Integer.toString(iter));
            //output sense labels from current iteration
            for(String line=in.readLine();line!=null;line=in.readLine()){
                //read a line
                String[]dict=line.split("\\s");
                int[]sent_previous_sense=new int[dict.length];
                if(iter!=0){
                    //if not the first iteration, we need read sense assignments from previous iteration
                    String line_previou_sense=in_sense.readLine();
                    //sense assignments for current sentence 
                    if(line_previou_sense==null)break;
                    if(!line.equals("")){
                        String[]string_line_sense=line_previou_sense.split("\\s");
                        for(int num=0;num<string_line_sense.length;num++)
                            sent_previous_sense[num]=Integer.parseInt(string_line_sense[num]);
                    }
                }

                if(line.equals("")){
                    if(Doc!=null){
                        DocList.add(Doc);
                        if(iter!=0){
                            DocSenseList.add(DocSense);
                        }
                    }
                    if(DocList.size()==batch_size){
                        counter+=batch_size;
                        alpha=Alpha*(Iter*n_doc-counter)/(Iter*n_doc);
                        for(int num=0;num<DocList.size();num++){
                            ArrayList<ArrayList<Integer>>this_doc=DocList.get(num);
                            ArrayList<int[]>this_doc_sense=null;
                            if(iter!=0)this_doc_sense=DocSenseList.get(num);
                            Callable<ArrayList<int[]> >worker=new MyCallable(this_doc,iter,this_doc_sense);
                            Future<ArrayList<int[]> >submit=executor.submit(worker);
                            list.add(submit);
                            //running each document in parallel
                        }
                        for(Future<ArrayList<int[]>>future : list){
                            ArrayList<int[]>new_doc_sense=future.get();
                            //store senses for current iteration
                            for(int[]new_sent_sense:new_doc_sense){
                                for(int num=0;num<new_sent_sense.length;num++){
                                    fw_new_sense.write(Integer.toString(new_sent_sense[num])+" ");
                                }
                                fw_new_sense.write("\n");
                            }
                            fw_new_sense.write("\n");
                        }
                        list=new ArrayList<Future<ArrayList<int[]> >>();
                        DocList=new ArrayList<ArrayList<ArrayList<Integer>>>();
                        if(iter!=0)DocSenseList=new ArrayList<ArrayList<int[]>>();
                    }
                    Doc=new ArrayList<ArrayList<Integer>>();
                    if(iter!=0)DocSense=new ArrayList<int[]>();
                    continue;
                }
                ArrayList<Integer>sen=new ArrayList<Integer>();
                for(int j=0;j<dict.length;j++){
                    int index=Integer.parseInt(dict[j]) ;
                    double l=1-Math.sqrt(0.0005/prob_word[index]);
                    double t=r.nextDouble();
                    if(t<l)sen.add(-1);
                    else sen.add(index);
                    //each line corresponds to one sentence. As in Mikolov's original paper, each token has a chance to be omitted.
                }
                Doc.add(sen);
                if(iter!=0)DocSense.add(sent_previous_sense);
                //discard sentences with less than 2 tokens
            }
            long end_time=System.currentTimeMillis();
            System.out.println((end_time-start_time)/1000);
            fw_new_sense.close();
        }
        executor.shutdown();
        Save(save_file);
    }
    public static class MyCallable implements Callable<ArrayList<int[]>>{
        ArrayList<ArrayList<Integer>>doc=null;
        double[]doc_vector;
        int iter;
        ArrayList<int[]>pre_doc_sense=null;
        public MyCallable(ArrayList<ArrayList<Integer>>doc,int iter,ArrayList<int[]>pre_doc_sense){
            this.doc=doc;
            this.doc_vector=new double[dimension];
            this.iter=iter;
            this.pre_doc_sense=pre_doc_sense;
        }
        public void get_doc_vector(){
            int length=0;
            double[]current_vector;
            for(ArrayList<Integer>sen:doc){
                for(int word:sen){
                    if(word==-1)continue;
                    length++;
                    doc_vector=my_math.plus(doc_vector,vect[word]);
                    //get doc-level embedding by using bag-of-word embeddings
                }
            }
            double p=1.000/length;
            doc_vector=my_math.dot(p,doc_vector);//averaged by number of tokens
        }
        public ArrayList<int[]> call()throws Exception {
            get_doc_vector();
            ArrayList<int[]>AllSenseIndex=new ArrayList<int[]>();
            int[]null_sense_list=null;
            for(int i=0;i<doc.size();i++){
                ArrayList<Integer>sen=doc.get(i);
                if (iter==0) AllSenseIndex.add(decent_sen(sen,null_sense_list));
                else AllSenseIndex.add(decent_sen(sen,pre_doc_sense.get(i)));
            }
            return AllSenseIndex;
        }

        public int[] decent_sen(ArrayList<Integer>sen,int[]previous_sense_index){
            double[]neu1e;
            double[]global_sen;
            double[]current_v;
            int[]label; label=new int[sen.size()];
            global_sen=new double[dimension];
            int length=0;
            for(int i=0;i<sen.size();i++){
                int word=sen.get(i);
                if(word==-1)continue;
                length++;
                global_sen=my_math.plus(global_sen,vect[word]);
            }
            //get sentence-level embedding by using bag-of-word embeddings
            double p=1.000/length;
            global_sen=my_math.dot(p,global_sen);
            
            HashMap<Integer,double[]>sense_List=null;
            //senses assigned to each token in current sentence
            HashMap<Integer,Integer>sen_table=null;
            for(int i=0;i<sen.size();i++){
                int time=0;
                double[]v;v=new double[dimension];
                
                int word=sen.get(i);
                if(word==-1){
                    //current token is omitted
                    label[i]=-1;continue;
                }
                vocab_word this_word=vocab.get(word); //get current token
                int half=1+(int)(word_window/2*r.nextDouble());//window size
                for(int win=0;win<half*2+1;win++){
                    if(win==half)continue;
                    int position=i-half+win;
                    if(position<0)continue;
                    if(position>=sen.size())continue;
                    int index=sen.get(position);//neighboring word index
                    if(index==-1)continue; //omitted token
                    neu1e=new double[dimension];
                    //hierarchical softmax, as in word2vect, predicting neighboring word given global embedding for current token
                    for(int j=0;j<this_word.point.size();j++){
                        int l2=this_word.point.get(j);
                        double f=0;
                        for(int k=0;k<dimension;k++)
                            f+=vect_t[l2][k]*vect[index][k];
                        double g=(-this_word.code.get(j)+my_math.sigmod(f))*alpha;
                        for(int k=0;k<dimension;k++)neu1e[k]+=g*vect_t[l2][k];
                        for(int k=0;k<dimension;k++)vect_t[l2][k]-=g*vect[index][k];
                    }
                    for(int k=0;k<dimension;k++)vect[index][k]-=neu1e[k];
                    if(iter==0&&position>=i)continue;
                    //hierarchical softmax, as in word2vect, but predicting senses for neighboring word given current global embedding for token
                    if(sense_match.get(index).size()!=1){
                        neu1e=new double[dimension];
                        int neighbor_sense;
                        if(position<i)neighbor_sense=label[position];
                        else neighbor_sense=previous_sense_index[position];
                        if(neighbor_sense==-1)continue;
                        double[]current_sense_v;
                        current_sense_v=sense_match.get(index).get(neighbor_sense);
                        for(int j=0;j<this_word.point.size();j++){
                            int l2=this_word.point.get(j);
                            double f=0;
                            for(int k=0;k<dimension;k++)
                                f+=vect_t[l2][k]*current_sense_v[k];
                            double p_=my_math.sigmod(f);
                            double g=(-this_word.code.get(j)+p_)*alpha;
                            for(int k=0;k<dimension;k++)neu1e[k]+=g*vect_t[l2][k];
                            //for(int k=0;k<dimension;k++)vect_t[l2][k]-=g*current_sense_v[k] ;
                        }
                        for(int k=0;k<dimension;k++)current_sense_v[k]-=neu1e[k];
                    }
                }

                //Chinese Restaurant Process
                sense_List=sense_match.get(word); //sense list for current token
                sen_table=cusomers_in_table.get(word);//sense table for current token
                int pre_index;
                if(iter!=0) {
                    pre_index=previous_sense_index[i];
                    if(pre_index!=-1)sen_table.put(pre_index,sen_table.get(pre_index)-1);
                }
                if(sense_List.size()==0){
                    //if no previous senses for current word, set up a new sense directly
                    sense_List.put(0,new double[dimension]);
                    sen_table.put(0,1);
                    label[i]=0; //store sense label for current word
                }
                else{
                    int should_new=-1; //whether we should set up a new sense
                    double[]prob;
                    if(sense_List.size()<20){
                        // maximum 20 senses for each word
                        prob=new double[sense_List.size()+1];
                        prob[sense_List.size()]=gamma;
                        should_new=1;
                    }
                    else {
                        // if more than 20 senses, we should not set up a new sense
                        prob=new double[sense_List.size()];
                        should_new=0;
                    }
                    double[]new_vector; //collecting information
                    new_vector=new double[dimension];
                    for(int j=0;j<prob.length-1;j++){
                        //iterating over each sense, computing probablity assigning current token to each sense
                        prob[j]=1;
                        time=0;
                        if(sense_List.size()==1)current_v=vect[word];
                        else current_v=sense_List.get(j);
                        prob[j]*=my_math.sigmod(my_math.dot(doc_vector,current_v));//evidence from global document vector
                        if(j==0)new_vector=my_math.plus(new_vector,doc_vector);
                        time++;
                        prob[j]*=my_math.sigmod(my_math.dot(global_sen,current_v));
                        if(j==0)new_vector=my_math.plus(new_vector,global_sen);//evidence from global sentence vector
                        time++;
                        for(int win=0;win<half*2+1;win++){
                            if(win==half)continue;
                            int position=i-half+win;
                            if(position<0)continue;
                            if(position>=sen.size())continue;
                            int index=sen.get(position);
                            if(index==-1)continue;
                            prob[j]*=my_math.sigmod(my_math.dot(vect[index],current_v));//evidence from global vector for neighboring tokens

                            if(j==0)new_vector=my_math.plus(new_vector,vect[index]);
                            time++;
                            if(iter==0&&position>i)continue;
                            if(sense_match.get(index).size()==1)continue;
                            if(!sense_match.get(index).containsKey(label[position]))continue;
                            int neighbor_sense_label;
                            if(position<i)neighbor_sense_label=label[position];
                            else neighbor_sense_label=previous_sense_index[position];
                            if(neighbor_sense_label==-1)continue;

                            double[]neighbor_sense_vector;
                            neighbor_sense_vector=sense_match.get(index).get(label[position]);
                            prob[j]*=my_math.sigmod(my_math.dot(neighbor_sense_vector,current_v)); //evidence from sense information from preceding tokens
                            if(j==0)new_vector=my_math.plus(new_vector,neighbor_sense_vector);
                            time++;
                        }
                        if(j==0)new_vector=my_math.dot(1.000/time,new_vector);
                    }
                    for(int j=0;j<prob.length-1;j++){
                        if(sen_table.get(j)<0){
                            prob[j]=0;System.out.println("negative apear!! Caution!!");
                        }
                        else prob[j]=sen_table.get(j)*Math.pow(prob[j],1.000/time);
                        //number of customers sitting at sense table by the probablity of assigning current token to that sense
                    }

                    for(int j=1;j<prob.length;j++){
                        prob[j]=prob[j-1]+prob[j];
                    }
                    double sample=r.nextDouble()*prob[prob.length-1];
                    //sampling sense label from Chinese restaurant problem
                    int new_label=-1;
                    for(int j=0;j<prob.length;j++){
                        if(sample<prob[j]){
                            new_label=j;
                            break;
                        }
                    }
                    label[i]=new_label;
                    //update parameters involved, both sense embeddings and gloabl word embeddings
                    if((new_label!=prob.length-1&&should_new==1)||should_new==0){
                        sen_table.put(new_label,sen_table.get(new_label)+1);
                        if(sense_List.size()!=1){
                            double[]current_sense_v;
                            current_sense_v=sense_List.get(new_label);
                            if(current_sense_v[0]==0&&current_sense_v[1]==0)System.out.println("problem");
                            double g;
                            g=my_math.sigmod(my_math.dot(doc_vector,current_sense_v));
                            g=g*(1-g)*alpha;
                            for(int k=0;k<dimension;k++)current_sense_v[k]-=g*doc_vector[k];
                            g=my_math.sigmod(my_math.dot(global_sen,current_sense_v));
                            g=g*(1-g)*alpha;
                            for(int k=0;k<dimension;k++)current_sense_v[k]-=g*global_sen[k];
                            for(int win=0;win<half*2+1;win++){
                                if(win==half)continue;
                                int position=i-half+win;
                                if(position<0)continue;
                                if(position>=sen.size())continue;
                                int index=sen.get(position);
                                if(index==-1)continue;
                                g=my_math.sigmod(my_math.dot(vect[index],current_sense_v));
                                //updating from the process of predicting sense label from neighoring embeddings
                                g=g*(1-g)*alpha;
                                for(int k=0;k<dimension;k++){
                                    current_sense_v[k]-=g*vect[index][k];
                                    //vect[index][k]-=g*current_sense_v[k];
                                }
                                if(iter==0&&position>=i)continue;
                                //updating from the process of predicting sense label from neighoring sense embeddings
                                if(sense_match.get(index).size()==1)continue;
                                if(!sense_match.get(index).containsKey(label[position]))continue;
                                
                                int neighbor_sense_label;
                                if(position<i)neighbor_sense_label=label[position];
                                else neighbor_sense_label=previous_sense_index[position];
                                if(neighbor_sense_label==-1)continue;

                                double[]temp_v;
                                temp_v=sense_match.get(index).get(label[position]);
                                g=my_math.sigmod(my_math.dot(temp_v,current_sense_v));
                                g=g*(1-g)*alpha;
                                for(int k=0;k<dimension;k++){
                                    current_sense_v[k]-=g*temp_v[k];
                                    temp_v[k]-=g*current_sense_v[k];
                                }
                            }
                        }
                    }
                    else{
                        //else if we encounter a new sense
                        if(sense_List.size()==1){
                            double[]v1;v1=new double[dimension];
                            for(int k=0;k<dimension;k++)v1[k]=vect[word][k];
                            sense_List.put(0,v1);
                        }
                        sense_List.put(prob.length-1,new_vector);
                        sen_table.put(prob.length-1,1);
                    }
                }
            }
            return label;
        }
    }

    public static void binary_tree(){
        //generating haffman tree, for details please refer to word2vect
        int min1i,min2i,i,a,b;
        double[]count; count=new double[prob_word.length*2-1];
        int[]parent;parent=new int[prob_word.length*2-1];
        int[]binary;binary=new int[prob_word.length*2-1];
        for(a=0;a<prob_word.length;a++)count[a]=prob_word[a];
        for(a=prob_word.length;a<2*prob_word.length-1;a++)count[a]=1000000;
        int vocab_size=prob_word.length;

        int pos1=prob_word.length-1;
        int pos2=prob_word.length;
        for(a=0;a<prob_word.length-1;a++){
            if(pos1>=0){
                if (count[pos1] < count[pos2]) {
                    min1i = pos1;pos1--;
                }
                else{
                    min1i = pos2; pos2++;
                }
            }
            else{
                min1i = pos2;pos2++;
            }
            if (pos1 >= 0) {
                if (count[pos1] < count[pos2]) {
                    min2i = pos1;pos1--;
                } else {
                    min2i = pos2;pos2++;
                }
            }else {
                min2i = pos2;pos2++;
            }
            count[vocab_size + a] = count[min1i] + count[min2i];
            parent[min1i]=vocab_size+a;
            parent[min2i]=vocab_size+a;
            binary[min2i]=1;
        }
        for (a = 0; a < vocab_size; a++) {
            vocab_word this_v=new vocab_word();
            b=a;
            ArrayList<Integer>T=new ArrayList<Integer>();
            T.add(b);
            while (true) {
                b = parent[b];
                if(b==0)break;
                T.add(b);
            }
            //System.out.println(T.size());
            for(i=T.size()-1;i>0;i--){
                int num=T.get(i);
                this_v.point.add(num-prob_word.length);
                this_v.code.add(binary[T.get(i-1)]);
            }
            vocab.add(this_v);
        }
    }
    public static void random(){
        //random initialization
        double epso=0.1;
        for(int i=0;i<vect.length;i++){
            for(int j=0;j<dimension;j++){
                vect[i][j]=(r.nextDouble()*2*epso-epso);
            }
        }
    }


    public static void Save(String FILE) throws Exception{
        //save file
        HashMap<Integer,double[]>match=null;
        HashMap<Integer,Integer>table=null;
        double[]store;
        //save sense specific embeddings
        FileWriter fw=new FileWriter(FILE+"_vect_sense");
        for(int i=0;i<vect.length;i++){
            match=sense_match.get(i);
            table=cusomers_in_table.get(i);
            if(match.size()!=1){
                int total=0;
                for(int index=0;index<match.size();index++)
                    total+=table.get(index);
                double[]prob;
                prob=new double[match.size()];
                for(int index=0;index<match.size();index++){
                    prob[index]=table.get(index)*1.000/total;
                }
                if(match.size()!=1){
                    fw.write("word "+Integer.toString(i)+" ");
                    int count=-1;
                    for(int index=0;index<match.size();index++){
                        if(!match.containsKey(index))continue;
                        if(table.get(index)*1.000/total<0.01)continue;
                        //ignore senses with less than occuring chance of 0.01
                        count++;
                        fw.write("sense"+Integer.toString(count)+" "+String.valueOf(table.get(index)*1.000/total)+"\n");
                        store=match.get(index);
                        for(int k=0;k<dimension;k++){
                            if(k!=dimension-1)
                                fw.write(store[k]+" ");
                            else
                                fw.write(store[k]+"\n");
                        }
                    }
                }
            }
        }
        fw.close();
        fw=new FileWriter(FILE+"_vect");
        //save global vector emebddings
        for(int i=0;i<vect.length;i++){
            for(int j=0;j<vect[0].length;j++){  
                if(j!=vect_t[0].length-1)
                    fw.write(vect[i][j]+",");
                else
                    fw.write(vect[i][j]+"\n");
            }
        }
        fw.close();
    }
    
    public static int num_of_docs(String filename)throws IOException {
    // compute number of docs
        BufferedReader in=new BufferedReader(new FileReader(filename));
        int n_line=0;
        for(String line=in.readLine();line!=null;line=in.readLine()){
            if(line.equals("")) n_line++;
        }
        return n_line;
    }

    public static int num_of_lines(String filename)throws IOException {
        //compute number of lines
        BufferedReader in=new BufferedReader(new FileReader(filename));
        int n_line=0;
        for(String line=in.readLine();line!=null;line=in.readLine()){
            n_line++;
        }
        in.close();
        return n_line;
    }

    public static void ReadFre(String filename)throws IOException {
    //read word frequency
        BufferedReader in=new BufferedReader(new FileReader(filename));
        int i=-1;
        int total=0;
        int word_num=0;
        for(String line=in.readLine();line!=null;line=in.readLine()){
            word_num++;
        }
        prob_word=new double[word_num];
        in=new BufferedReader(new FileReader(filename));
        for(String line=in.readLine();line!=null;line=in.readLine()){
            i++;
            String[]dict=line.split("\\s");
            prob_word[i]=Double.parseDouble(dict[0]);
        }
        for(i=0;i<word_num;i++){
            HashMap<Integer,double[]>t_1=new HashMap<Integer,double[]>();
            sense_match.put(i,t_1);
            HashMap<Integer,Integer>t_2=new HashMap<Integer,Integer>();
            cusomers_in_table.put(i,t_2);
        }
    }
    public static void printArrayList(ArrayList<Integer>A){
        String string="";
        for(int i=0;i<A.size();i++)
            string=string+Integer.toString(A.get(i))+" ";
        System.out.println(string);
    }
    public static void printArray(int[]A){
        String string="";
        for(int i=0;i<A.length;i++)
            string=string+Integer.toString(A[i])+" ";
        System.out.println(string);
    }
}

class vocab_word{
    ArrayList<Integer>point=new ArrayList<Integer>();
    ArrayList<Integer>code=new ArrayList<Integer>();
}
