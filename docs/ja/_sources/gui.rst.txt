******************************
GUI版
******************************

簡単にNIMS-OSを実行できるように，GUIバージョンを作成し公開しています．


インストール
===========================


動作環境
-----------------------

* Windows 10 (64bit)

* Windows 11




インストール手順
-------------------------

* GUI版をインストールする前に，Python版NIMS-OSをあらかじめ :doc:`インストール<./install>` してください．



* GitHubに掲載されているインストーラ(nimsos-gui-main.zip)を `GitHub <https://github.com/nimsos-dev/nimsos-gui>`_ よりダウンロードします．


* 適当な場所でzipを解凍すると，下記のフォルダ構成になっています．SetupフォルダにWindows用のセットアッププログラムが入っています．

.. image:: ../img/gui_install_1.png
   :align: center
   :scale: 25%



|

* Setupフォルダ内にあるSetup.exeを起動させると以下のようにインストーラーが起動します．次へをクリックします．

.. image:: ../img/gui_install_2.png
   :align: center
   :scale: 35%



|

* インストール先を指定する画面に移ります．ここでは，D:¥NIMSOSフォルダを指定してください．もしDドライブが無い場合，CドライブにNIMSOSフォルダを作成し，そのフォルダを指定します．Cドライブにインストールした場合，インストール後に，NIMSOSフォルダ内にあるNIMSOS.iniをNotepadで開き，Dドライブで指定されている箇所をCドライブに変更する必要があります．

.. image:: ../img/gui_install_3.png
   :align: center
   :scale: 35%



|

* 以下の画面で，次をクリックするとインストールがスタートします．


.. image:: ../img/gui_install_4.png
   :align: center
   :scale: 35%



|


* インストールが正常終了すると，以下の画面に切り替わります．閉じるをクリックするとインストールは終了です．

.. image:: ../img/gui_install_5.png
   :align: center
   :scale: 35%



|


* インストール終了後，スタートメニューにNIMS-OSが追加されている事を確認して下さい．

.. image:: ../img/gui_install_6.png
   :align: center
   :scale: 35%



|

* インストール先のNIMS0S内には，SCRIPTフォルダがあり，その中に以下のPythonスクリプトが入っています．これらのスクリプトがGUIでは実行されます．

.. image:: ../img/gui_install_7.png
   :align: center
   :scale: 55%



|


1. ``ai_tool.py`` : 人工知能の実行用スクリプト

#. ``ai_tool_original.py`` :	自作人工知能の実行用スクリプト(originalを選択した際に実行されます．)

#. ``analysis_output.py`` : ロボット実験結果の解析と実験条件ファイルの更新用スクリプト

#. ``analysis_output_original.py`` : 自作ロボット実験結果の解析と実験条件ファイルの更新用スクリプト(originalを選択した際に実行されます．)

#. ``preparation_input.py`` :	ロボット実験用インプットファイルの作成およびロボット実験の実行用スクリプト

#. ``preparation_input_original.py`` :	自作ロボット実験用インプットファイルの作成およびロボット実験の実行用スクリプト(originalを選択した際に実行されます．)


アンインストール
===========================


* アンインストールする場合，Windowsのコントロールパネル → プログラム → プログラムと機能に移動し，NIMS-OSを以下のようにアンインストールしてください．

.. image:: ../img/gui_uninstall.png
   :align: center
   :scale: 35%



|


使い方
====================

GUIバージョンの操作画面は下図のようになっています．

.. image:: ../img/gui_operation.png
   :align: center


|

このGUIバージョンでは， ``candidates_file`` の名前は ``candidates.csv`` に， ``proposals_file`` の名前は， ``proposals.csv`` にそれぞれ固定されていることに注意してください．
実行手順について以下に紹介します．

1. Parametersの部分にnumber of objectives, proposals, and cyclesのそれぞれの値を入力します．

#. AI algorithmの部分で使用する手法を選択します．新しく独自作成したモジュール ``ai_tool_original.py`` を利用する場合は，Originalをクリックしてください．

#. Robotic systemでは，ロボット実験を選択します．新しく独自作成したモジュール ``preparation_input_original.py`` および ``analysis_output_original.py`` を利用する場合は，Originalをクリックしてください．

#. Controllerのrunボタンを押すことで，自動材料探索がスタートします．

実行すると，Cycle counterが動作し，何サイクル目の実験を行なっているかがわかります．
また，AI algorithmの実行にかかる時間，ロボット実験にかかる時間が計測されます．
それに準じて，残り予定時間が出力されるようになっています．
さらに，ResultsウィンドウにPython版の標準出力がリアルタイムで表示されるため，現状の目的関数値を知ることができます．
このResultsに出力される結果は，Outputボタンを押すことで，ファイルとして保存することもできます．


* 自動探索を一時停止したい場合はControllerウィンドウのstopボタンを押すことで， ``candidates_file`` がアップデートされたタイミングで探索が停止します．

* 設定をリセットしたい場合，Controllerのresetボタンを押すことでリセットされます．

