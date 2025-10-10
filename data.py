def create_data_module(args, tokenizer):
    if args.model in ['llm4rec']:
        from utils.LLM4Rec_data import LLM4RecDataModule
        datamodule = LLM4RecDataModule(args, tokenizer)
    return datamodule