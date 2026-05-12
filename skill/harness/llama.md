# 存储管理
- 大文件写入到/tmp/shared-storage/ExpertGym/LLaMA/，分类管理，可以软连接到项目主目录

- 环境暂时使用conda activate BFCL，如果后续需要装其他库，可以 copy 一份到大文件目录，然后在共享盘 内的本地路径装。
# Expert 模型族下载目录
Llama-3.2-3B-Instruction Series To verify generalization across architectures, we utilize the following agents based on Llama-3.2-3B-Instruct:
Base Model (Llama-3.2-3B-Instruct) (Grattafiori
et al., 2024)
Llama-3.2-3B-Instruct
Math Agent (GRPO-Math) (Guo et al., 2025)
Llama-3.2-3B-Instruct-GRPO-MATH-1EPOCH
Tool Agent (ToolRL) (Qian et al., 2025)
ToolRL-Llama3.2-3B
Search Agent (ZeroSearch) (Sun et al., 2025)
ZeroSearch_google_V2_Llama_3.2_3B_Instruct
# github仓库-用来找各个expert RL的原始训练数据
