def train_with_adaptive_dpc(model, train_loader, val_loader, criterion, optimizer, 
                          scheduler, num_epochs=100, initial_bits=8, min_bits=4):
    # Initialize Adaptive DPC
    adpc = AdaptiveDPC(model, initial_bits, min_bits)
    
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Dictionary to store gradient statistics
        gradient_stats = {}
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with quantized weights
            adpc.quantize_weights()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Collect gradient statistics before optimizer step
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name and param.grad is not None:
                        layer_name = name.replace('.weight', '')
                        grad_norm = param.grad.norm().item()
                        grad_var = param.grad.var().item()
                        
                        if layer_name not in gradient_stats:
                            gradient_stats[layer_name] = {'norm': [], 'var': []}
                        gradient_stats[layer_name]['norm'].append(grad_norm)
                        gradient_stats[layer_name]['var'].append(grad_var)
            
            optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # Update precision based on gradient statistics
        avg_gradient_stats = {}
        for name in gradient_stats:
            avg_gradient_stats[name] = {
                'norm': np.mean(gradient_stats[name]['norm']),
                'var': np.mean(gradient_stats[name]['var'])
            }
        adpc.update_precision(avg_gradient_stats)
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, adpc)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print current bit-width distribution
        print_bit_width_distribution(adpc)
    
    return model

def validate(model, val_loader, criterion, adpc):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass with quantized weights
            adpc.quantize_weights()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_corrects.double() / len(val_loader.dataset)
    
    return val_loss, val_acc

def print_bit_width_distribution(adpc):
    bit_counts = {}
    for name, bits in adpc.bit_widths.items():
        if bits not in bit_counts:
            bit_counts[bits] = 0
        bit_counts[bits] += 1
    
    print("Bit-width distribution:")
    for bits in sorted(bit_counts.keys()):
        print(f"{bits}-bit: {bit_counts[bits]} layers")