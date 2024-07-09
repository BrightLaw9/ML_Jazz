import logging
import os
import re
import glob
from urllib.parse import urlencode
from urllib.request import urlopen

import yt_dlp as yt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


def _filter_video(video_info, include, exclude) -> bool:
    """Return True if video should be skipped.

    If video lacks a required include term in its metadata, skip it.

    If video has any required exclude term in its metadata, skip it.
    """
    title = video_info["title"]
    description = video_info["description"]
    tags = video_info["tags"]
    categories = video_info["categories"]
    metadata = [title, description, *tags, *categories]
    haystack = " ".join(metadata).lower()

    for file in glob.glob(os.getcwd() + ("/*")): 
        if file.split("\\")[-1].startswith(title):
            logger.info(f"**Skipping {title}, all ready downloaded**")
            return True # Don't include

    if include:
        if all(w not in haystack for w in include):
            return True

    if exclude:
        if any(w in haystack for w in exclude):
            return True
    return False


def scrape(query, include, exclude, quiet, verbose, overwrite, limit):
    """Search YouTube and download audio from discovered videos."""

    # Search YouTube for videos.
    query_string = urlencode({"search_query": f"{query} {include}"})
    url = f"http://youtube.com/results?{query_string}"

    # Get video IDs from search results.
    with urlopen(url) as response:
        html = response.read().decode("utf-8")
        logger.debug(html)

    # Search for video IDs in HTML response.
    video_ids = re.findall(r"\"\/watch\?v=(.{11})", html)

    # Go through each video ID and download audio.
    for video_id in video_ids[:limit]:
        # Construct video URL.
        logger.info(f"Getting video: {video_id}")
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Always prefer highest quality audio.
        download_options = {
            "format": "bestaudio/best",
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            "verbose": verbose,
            "quiet": quiet,
            "nooverwrites": not overwrite,
            "writeinfojson": True,
            "writethumbnail": True,
            "writedescription": True,
        }
        ydl = yt.YoutubeDL(download_options)

        # Fetch metadata.
        video_info = ydl.extract_info(video_url, download=False)
        logger.debug(video_info)

        # TODO Use builtin functionality in youtube-dl for this instead.
        # Inspect video metadata to determine if video should be skipped.
        if _filter_video(video_info, include, exclude):
            continue

        # Download audio.
        ydl.download([video_url])


if __name__ == "__main__": 
    scrape("Jazz", ["bebop"], [], False, False, False, 100); # Last param is limit number of downloads