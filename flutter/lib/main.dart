import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path/path.dart' show join;
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  final TextEditingController _inputController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Home Page'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _inputController,
              decoration: InputDecoration(labelText: 'Enter a name'),
            ),
            SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => CameraScreen(
                      imageName: _inputController.text,
                    ),
                  ),
                );
              },
              child: Text('Confirm'),
            ),
          ],
        ),
      ),
    );
  }
}
class CameraScreen extends StatefulWidget {
  final String imageName;

  CameraScreen({required this.imageName});

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  XFile? _capturedImage;
  int _currentSilhouetteIndex = 0;
  bool _isFlashOn = false;
  String position = "front";
  List<String> _silhouetteImages = [
    'assets/front.png',
    'assets/side.png',
    'assets/r_tuck.png',
    'assets/tuck.png',
    'assets/pike.png',
  ];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final camera = cameras.first;

    _controller = CameraController(
      camera,
      ResolutionPreset.max,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    
    _initializeControllerFuture = _controller.initialize();

    _initializeControllerFuture.then((_) {
      if (mounted) {
        setState(() {});
      }
    });
    await _controller.setFlashMode(FlashMode.auto);

}
  void _deleteImage() async {
    if (_capturedImage != null) {
      final File imageFile = File(_capturedImage!.path);
      if (await imageFile.exists()) {
        await imageFile.delete();
        setState(() {
          _capturedImage = null;
        });
      }
    }
  }

  void _toggleSilhouette() {
    setState(() {
      _currentSilhouetteIndex =
          (_currentSilhouetteIndex + 1) % _silhouetteImages.length;
      List<String> parts = _silhouetteImages[_currentSilhouetteIndex].split('/');
      String res = parts.last;
      position = res.replaceAll('.png', '');
    });
  }
  void _toggleFlash() {
    if (_controller.value.isInitialized) {
      setState(() {
        _isFlashOn = !_isFlashOn;
        _controller.setFlashMode(
          _isFlashOn ? FlashMode.torch : FlashMode.off,
        );
      });
    }
  }
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    double previewWidth = 384 * 0.62;
    double previewWidthLittleSquares = 384 * 0.075;

    return Scaffold(
      appBar: AppBar(
        title: Text(''),
        leading: IconButton(
          icon: Image.asset(
            'assets/home_page.png',
            width: 24.0,
            height: 24.0,
          ),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                CameraPreview(_controller),
                if (_capturedImage != null)
                  Positioned.fill(
                    child: Image.file(
                      File(_capturedImage!.path),
                      fit: BoxFit.cover,
                    ),
                  ),
                Positioned(
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  child: Center(
                    child: Container(
                      width: previewWidth,
                      height: previewWidth,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.red, width: 2.0),
                      ),
                      child: Stack(
                        children: [
                          Positioned(
                            top: 0,
                            left: 0,
                            child: Container(
                              width: previewWidthLittleSquares,
                              height: previewWidthLittleSquares,
                              decoration: BoxDecoration(
                                color: Colors.blue.withOpacity(0.3),
                                border: Border.all(color: Colors.blue, width: 2.0),
                              ),
                            ),
                          ),
                          Positioned(
                            top: 0,
                            right: 0,
                            child: Container(
                              width: previewWidthLittleSquares,
                              height: previewWidthLittleSquares,
                              decoration: BoxDecoration(
                                color: Colors.blue.withOpacity(0.3),
                                border: Border.all(color: Colors.blue, width: 2.0),
                              ),
                            ),
                          ),
                          Positioned(
                            bottom: 0,
                            left: 0,
                            child: Container(
                              width: previewWidthLittleSquares,
                              height: previewWidthLittleSquares,
                              decoration: BoxDecoration(
                                color: Colors.blue.withOpacity(0.3),
                                border: Border.all(color: Colors.blue, width: 2.0),
                              ),
                            ),
                          ),
                          Positioned(
                            bottom: 0,
                            right: 0,
                            child: Container(
                              width: previewWidthLittleSquares,
                              height: previewWidthLittleSquares,
                              decoration: BoxDecoration(
                                color: Colors.blue.withOpacity(0.3),
                                border: Border.all(color: Colors.blue, width: 2.0),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                Positioned.fill(
                  child: Image.asset(
                    _silhouetteImages[_currentSilhouetteIndex],
                  ),
                ),
              ],
            ),

          ),
          if (_capturedImage != null)
            Container(
              padding: EdgeInsets.all(5.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: () {
                      _deleteImage();
                    },
                    child: Text('Delete'),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      _toggleSilhouette();
                      setState(() {
                        _capturedImage = null;
                      });
                    },
                    child: Text('Ok'),
                  ),
                  SizedBox(width: 24.0),
                  FloatingActionButton(
                  onPressed: _toggleFlash,
                  child: Image.asset(
                    'assets/flashlight.jpg',
                    width: 24.0,
                    height: 24.0,
                    color: _isFlashOn ? Colors.white : Colors.black,
                    ),
                  ),
                ],
              ),
            )
          else
            Container(
              padding: EdgeInsets.all(5.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Expanded(
                    child: Center(
                        child: FloatingActionButton(
                          onPressed: () async {
                            try {
                            await _initializeControllerFuture;
                            final XFile picture = await _controller.takePicture();
                            final String directory =
                                (await getExternalStorageDirectory())?.path ?? "";
                            final String formattedTimestamp =
                                DateFormat('yyyy-MM-dd_HHmmss').format(DateTime.now());
                            final String fileName =
                                '${widget.imageName}_${position}_$formattedTimestamp.jpg';
                            final String filePath = join(directory, fileName);
                            await picture.saveTo(filePath);
                            setState(() {
                              _capturedImage = XFile(filePath);
                            });
                          } catch (e) {
                            print(e);
                          }
                        },
                        child: IconTheme(
                          data: IconThemeData(size: 24),
                          child: Image.asset(
                            'assets/camera_button.png',
                          ),
                        ),
                      ),
                    ),
                  ),
                  FloatingActionButton(
                    onPressed: _toggleFlash,
                    child: Image.asset(
                      'assets/flashlight.jpg',
                      width: 24.0,
                      height: 24.0,
                      color: _isFlashOn ? Colors.white : Colors.black,
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
