# LLM4FS

The code in this toolbox implements "LLM4FS: Leveraging Large Language Models for Feature Selection and How to Improve It". 


### Testing
For the LLM-based method, in the LLMBased_demo folder, run LLMBased_demo.py, and then run classical.py to obtain the performance test results. 

For the hybrid strategy we proposed, in the HybridStrategy_demo folder, first upload the description_LLM4OPT.txt file and the Credit-G_200.csv file in the prompt file to the LLMs. Then, put the json file given by the LLMs into the file with the path HybridStrategy_demo/data/classical/Credit-G/deepseekR1+RandomForest_output.json. Finally, run classical.py to obtain the performance test results. 


### Citation
Please give credits to this paper if this code is useful and helpful for your research.
