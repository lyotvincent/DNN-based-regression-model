

def figures(history,figure_name="plots"):
    from keras.callbacks import History
    if isinstance(history,History):
        import matplotlib.pyplot as plt
        hist     = history.history
        epoch    = history.epoch
        # acc      = hist['acc']
        # loss     = hist['loss']
        # val_loss = hist['val_loss']
        # val_acc  = hist['val_acc']
        acc      = hist['r_square']
        loss     = hist['loss']
        val_loss = hist['val_loss']
        val_acc  = hist['val_r_square']
        plt.figure(1)

        plt.subplot(221)
        plt.plot(epoch,acc)
        # plt.title("Training accuracy vs Epoch")
        plt.title("Training R_square vs Epoch")
        plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        plt.ylabel("r_square")

        plt.subplot(222)
        plt.plot(epoch,loss)
        plt.title("Training loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(223)
        plt.plot(epoch,val_acc)
        # plt.title("Validation Acc vs Epoch")
        plt.title("Testing R_square vs Epoch")
        plt.xlabel("Epoch")
        # plt.ylabel("Validation Accuracy")
        plt.ylabel("Testing R_square")

        plt.subplot(224)
        plt.plot(epoch,val_loss)
        plt.title("Testing loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Testing Loss")
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.show()
    else:
        print("Input Argument is not an instance of class History")
