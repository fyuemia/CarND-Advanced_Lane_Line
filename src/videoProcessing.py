from moviepy.editor import VideoFileClip
import os

import src.imagePipeline as imgPipeline


def gen_video_tracker(video, subclip=False, debug_window=False):
    '''
    handle the project_video use pipeline function
    '''
    # Create pipeline instance
    print("build pipeline instance...")
    left = imgPipeline.Line()
    right = imgPipeline.Line()
    pipeline = imgPipeline.Pipeline(left, right)

    # checkif debug_window if turn on
    pipeline.debug_window = True if debug_window else False

    # check if handle the subclip
    if subclip:
        print("test on 5 second video")
        clip = VideoFileClip("../" + video).subclip(0, 5)
    else:
        print("handle the whole video")
        clip = VideoFileClip("../" + video)

    # choice the pipeline according the video
    if video == "project_video.mp4" or video == "challenge_video.mp4" or video == "harder_challenge_video.mp4":
        white_clip = clip.fl_image(pipeline.pipeline)
    else:
        print("Wrong Vidoe Name, please check the video name!!!")
        return 1

    white_output = "../output_video/" + video
    white_clip.write_videofile(white_output, audio=False)

    # write the information to the consel
    print("processed {} images".format(pipeline.image_counter))
    print("Search Failure: {}".format(pipeline.search_fail_counter))
    print("The video is at ./output_video/temp/")



def get_image(video, dst, frame_list):
    '''
    get image from the video
    frame_list = [1, 3, 5, 7, 9]
    '''
    clip = VideoFileClip(video)

    for t in frame_list:
        imgpath = os.path.join(dst, '{}.jpg'.format(t))
        clip.save_frame(imgpath, t)


if __name__ == "__main__":
    """
    choise one line to uncoment and run the file, gen the video.
    the video will be output to ./outpu_videos/temp/
    option: subclip = True, just use (0-5) second video, False, use total long video.
    option: debug_window = True, project the debug window on the up-right corner of the screen to visualize the image handle process
                                and write the fit lane failure/search lane failure image to ./output_videos/temp/images
    """
    get_image("../challenge_video.mp4", "../project_images/challenge/", [i for i in range(1,30)])
    get_image("../harder_challenge_video.mp4", "../project_images/harder_challenge/", [i for i in range(1,50)])

    # gen_video_tracker("project_video.mp4", subclip=False, debug_window=True)
    # gen_video_tracker("challenge_video.mp4", subclip=False, debug_window=True)
    # gen_video_tracker("harder_challenge_video.mp4", subclip=False, debug_window=True)
