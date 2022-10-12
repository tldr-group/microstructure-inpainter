from run_gui import *
try:
    import pyi_splash
    pyi_splash.update_text("Loading inpainter...")
    if __name__ == '__main__':
        pyi_splash.close()
        main()
except:
    if __name__ == '__main__':
        main()




