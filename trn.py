import torch 
from main import * 

def runValidation(model, validation_ds, tokenizer_en, tokenizer_te, max_seq_len, device):
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device) 

            cls_id = tokenizer_te.token_to_id('[CLS]')
            sep_id = tokenizer_te.token_to_id('[SEP]')

            # Computing the output of the encoder for the source sequence
            encoder_output = model.encode(encoder_input, encoder_mask)
            # for prediction task, the first token that goes in decoder input is the [CLS] token
            decoder_input = torch.empty(1, 1).fill_(cls_id).type_as(encoder_input).to(device)
            # since we need to keep adding the output back to the input until the [SEP] - end token is received.
            while True:
                # check if the max length is received
                if decoder_input.size(1) == max_seq_len:
                    break

                # recreate mask each time the new output is added the decoder input for next token prediction
                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

                # apply projection only to the next token
                out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                # apply projection only to the next token  
                prob = model.project(out[:, -1])

                # select the token with highest probablity which is a greedy search implementation
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat(
                    [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
                )
                # check if the new token is the end of token
                if next_word == sep_id:
                    break
            # final output is the concatinated decoder input till the end token is reached
            model_out = decoder_input.squeeze(0)

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_te.decode(model_out.detach().cpu().numpy())

            # Print the source, target and model output
            print('-'*55)
            # print_msg(f"{f'SOURCE: ':>12}{source_text}")
            # print_msg(f"{f'TARGET: ':>12}{target_text}")
            # print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
            print(f'Source Text: {source_text}')
            print(f'Target Text: {target_text}')
            print(f'Predicted by MalayGPT: {model_out_text}')

            if count == 2:
                break

def train_model(preload_epoch = None):
    epochs = 100 
    initial_epoch = 0 
    global_step = 0 

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, eps = 1e-9)

    if preload_epoch is not None:
        model_filename = f"./TeluguGPT/model_{preload_epoch}.pt"
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'], strict=False)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'], strict=False) 
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_english.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, epochs):
        # torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            target_label = batch['target_label'].to(device) # (B, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            projection_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(projection_output.view(-1, tokenizer_telugu.get_vocab_size()), target_label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) 

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # VALIDATION BLOCK STARTS HERE [Runs every epoch after the training block is complete]
        runValidation(model, validation_dataloader, tokenizer_english, tokenizer_telugu, max_seq_length, device)

        # Save the model at the end of every epoch
        model_filename = f"./TeluguGPT/model_{epoch}.pt" 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

train_model(preload_epoch = None)  