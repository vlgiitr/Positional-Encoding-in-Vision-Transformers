def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, step, use_wandb=False, print_freq=100):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            elif torch.backends.mps.is_available():
                images = images.to('mps')
                target = target.to('mps')

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

    progress.display_summary()
    
    if use_wandb:        
        log_data = {
            'val/loss': losses.avg,
            'val/acc@1': top1.avg,
            'val/acc@5': top5.avg,
        }
        wandb.log(log_data, step=step)

    return top1.avg

def train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device):
    
    print("starting")
    
    def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_step = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Loaded checkpoint. Resuming from step {start_step}")
        return start_step, best_acc1
    
    # Load checkpoint
    start_step, best_acc1 = load_checkpoint("/kaggle/input/pmochina/BaseLine_VIT.pt", original_model, optimizer, scheduler)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    print_freq = 100
    log_steps = 2500
    
    progress = ProgressMeter(
        total_steps,
        [batch_time, data_time, losses, top1, top5]
    )

    model.train()
    end = time.time()
    
    def infinite_loader():
        while True:
            yield from train_loader
            
    for step, (images, labels_a, labels_b, lam) in zip(range(start_step + 1, total_steps + 1), infinite_loader()):
        
        print(step)
        
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        labels_a = labels_a.to(device, non_blocking=True)
        labels_b = labels_b.to(device, non_blocking=True)
        
        # Convert lam to a tensor if it's not already one
        if not isinstance(lam, torch.Tensor):
            lam = torch.tensor(lam, device=device)
        else:
            lam = lam.to(device, non_blocking=True)

        output = model(images)
        loss = lam * criterion(output, labels_a) + (1 - lam) * criterion(output, labels_b)

        # Compute accuracy (this is an approximation for mixed labels)
        acc1_a, acc5_a = accuracy(output, labels_a, topk=(1, 5))
        acc1_b, acc5_b = accuracy(output, labels_b, topk=(1, 5))
        acc1 = lam * acc1_a + (1 - lam) * acc1_b
        acc5 = lam * acc5_a + (1 - lam) * acc5_b

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        loss.backward()
        l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if step % print_freq == 0:
            progress.display(step)
            if wandb:
                with torch.no_grad():
                    l2_params = sum(p.square().sum().item() for _, p in model.named_parameters())
                    
                samples_per_second_per_gpu = images.size(0) / batch_time.val
                samples_per_second = samples_per_second_per_gpu 
                log_data = {
                    "train/loss": losses.val,
                    'train/acc@1': top1.val,
                    'train/acc@5': top5.val,
                    "data_time": data_time.val,
                    "batch_time": batch_time.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": scheduler.get_last_lr()[0],
                    "l2_grads": l2_grads.item(),
                    "l2_params": math.sqrt(l2_params)
                }
                wandb.log(log_data, step=step)
        
        if ((step % print_freq == 0) and ((step % log_steps != 0) and (step != total_steps))):        
            save_checkpoint_step(step, model, best_acc1, optimizer, scheduler, checkpoint_path)
                
        if step % log_steps == 0:
            acc1 = validate(val_loader, original_model, criterion, step)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint({
                'step': step,
                'state_dict': original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, checkpoint_path)
            
        elif step == total_steps:
            acc1 = validate(val_loader, original_model, criterion, step)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint({
                'step': step,
                'state_dict': original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, checkpoint_path)
            
        if step % 20000 == 0 and step > 0:
            break

        scheduler.step()

# Use the modified train function
train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device)
